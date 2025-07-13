import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import math

class FedAvGHEAL(FederatedModel):
    NAME = 'fedavgheal'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedAvGHEAL, self).__init__(nets_list,args,transform)
        
        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}
        
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}

        ######### 
        self.client_stats = {
            m: {"total_reward": 0.0, "num_selections": 0}
            for m in range(len(self.nets_list))
        }

        # Placeholder for Q-scores (trust/quality scores), assumed to be updated elsewhere
        self.q_score = {
            m: 1.0 for m in range(len(self.nets_list))
        }  # Initialize with default reward of 1.0

        # Check if 'clients_per_round' is set, fallback to all clients
        if not hasattr(self.args, 'clients_per_round'):
            self.args.clients_per_round = len(self.nets_list)

        self.dynamic_clients_per_round = self.args.clients_per_round
        self.exploration_weight = 1.0  # Starts fully exploring
        self.previous_mean_acc = 0
        self.gain_threshold = 1.5  # Tune this based on your dataset
        self.min_clients = max(3, self.args.parti_num // 4)
        self.max_clients = self.args.parti_num

        self.val_acc_before = {}
        self.val_acc_after = {}

        

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    
    def compute_reward(self, client_id):
        # Example: negative of L2 norm of the update (smaller = better)
        update = self.client_update[client_id]
        norm = torch.sqrt(sum([torch.norm(p.float())**2 for p in update.values()]))
        reward = 1.0 / (norm + 1e-6)  # Avoid division by 0
        return reward

    # # Add this to the class (outside any other function)
    # def select_clients_ucb(self, current_round):
    #     ucb_scores = []
    #     total_rounds = current_round + 1

    #     print(f"\n--- Round {total_rounds} UCB Scores ---")

    #     for client_id in range(len(self.nets_list)):
    #         stats = self.client_stats[client_id]
    #         num_selections = stats["num_selections"]
    #         total_reward = stats["total_reward"]
    #         avg_reward = total_reward / num_selections if num_selections > 0 else 0

    #         if stats["num_selections"] == 0:
    #             ucb = float('inf')
    #         else:
    #             ucb = avg_reward + np.sqrt(2 * np.log(total_rounds) / stats["num_selections"])

    #         ucb_scores.append((ucb, client_id))

    #         q_score_val = self.q_score[client_id] if client_id in self.q_score else 0.0
    #         print(f"Client {client_id} | Q-Score: {q_score_val:.4f} | Selections: {num_selections} | UCB: {ucb:.4f}")

    #     ucb_scores.sort(reverse=True)
    #     selected = [client_id for _, client_id in ucb_scores[:self.args.clients_per_round]]
        
    #     print(f"Selected Clients for this round: {selected}\n")
    #     return selected

    # def select_clients_ucb(self, current_round):
    #     ucb_scores = []
    #     total_rounds = current_round + 1

    #     # Decay exploration weight
    #     self.exploration_weight = max(1.0 - current_round / 20.0, 0.1)

    #     print(f"\n--- Round {current_round + 1} UCB Scores ---")
    #     for client_id in range(len(self.nets_list)):
    #         stats = self.client_stats[client_id]
    #         avg_reward = stats["total_reward"] / stats["num_selections"] if stats["num_selections"] > 0 else 0
    #         # ucb = avg_reward + self.exploration_weight * np.sqrt(2 * np.log(total_rounds) / (stats["num_selections"] + 1e-6))
    #         if stats["num_selections"] == 0:
    #             ucb = float('inf')
    #         else:
    #             ucb = avg_reward + np.sqrt(2 * np.log(total_rounds) / stats["num_selections"])

    #         ucb_scores.append((ucb, client_id))
    #         print(f"Client {client_id} | Q-Score: {self.q_score[client_id]:.4f} | Selections: {stats['num_selections']} | UCB: {ucb:.4f}")

    #     ucb_scores.sort(reverse=True)

    #     # Bound the client count within [min_clients, max_clients]
    #     self.dynamic_clients_per_round = min(max(self.dynamic_clients_per_round, self.min_clients), self.max_clients)

    #     selected = [client_id for _, client_id in ucb_scores[:self.dynamic_clients_per_round]]
    #     print(f"Selected Clients for this round: {selected}")
    #     return selected

    def select_clients_ucb(self, current_round):
        ucb_scores = []
        total_rounds = current_round + 1

        # Decay exploration rate epsilon
        epsilon = max(0.1, 1.0 - current_round / 50.0)  # slowly decays to 0.1

        print(f"\n--- Round {total_rounds} UCB Scores ---")

        for client_id in range(len(self.nets_list)):
            stats = self.client_stats[client_id]
            avg_reward = stats["total_reward"] / stats["num_selections"] if stats["num_selections"] > 0 else 0

            if stats["num_selections"] == 0:
                ucb = float('inf')  # Encourage exploration
            else:
                ucb = avg_reward + np.sqrt(2 * np.log(total_rounds) / stats["num_selections"])

            ucb_scores.append((ucb, client_id))

        ucb_scores.sort(reverse=True)
        selected = []

        num_explore = int(self.dynamic_clients_per_round * epsilon)
        num_exploit = self.dynamic_clients_per_round - num_explore

        # Exploration: randomly pick from clients who aren't in top
        unexplored_clients = list(set(range(len(self.nets_list))) - set([cid for _, cid in ucb_scores[:num_exploit]]))
        explore_clients = np.random.choice(unexplored_clients, min(num_explore, len(unexplored_clients)), replace=False).tolist()

        # Exploitation: pick top clients via UCB
        exploit_clients = [client_id for _, client_id in ucb_scores[:num_exploit]]

        selected = exploit_clients + explore_clients
        print(f"Selected Clients (Exploit={len(exploit_clients)} Explore={len(explore_clients)}): {selected}\n")
        return selected

    

    def adjust_clients_per_round(self, current_mean_acc):
        gain = current_mean_acc - self.previous_mean_acc
        print(f"Accuracy Gain since last round: {gain:.4f}")

        if gain < self.gain_threshold:
            # Performance stagnant → explore more
            self.dynamic_clients_per_round = min(self.dynamic_clients_per_round + 1, self.max_clients)
        else:
            # Doing well → exploit top clients
            self.dynamic_clients_per_round = max(self.dynamic_clients_per_round - 1, self.min_clients)

        print(f"Updated dynamic_clients_per_round = {self.dynamic_clients_per_round}")
        self.previous_mean_acc = current_mean_acc


    

    def loc_update(self, priloader_list):
        # Select top-K clients using UCB
        def compute_ucb(client_id, round_t):
            stats = self.client_stats[client_id]
            if stats["num_selections"] == 0:
                return float("inf")  # Encourage exploration
            avg_reward = stats["total_reward"] / stats["num_selections"]
            exploration_term = math.sqrt(2 * math.log(round_t + 1) / stats["num_selections"])
            return avg_reward + exploration_term

       


        # Number of clients to select this round
        # K = self.args.clients_per_round
        K = self.args.clients_per_round or self.args.parti_num 
        # Round index
        round_t = self.epoch_index

        # Select K clients based on UCB
        self.online_clients = self.select_clients_ucb(round_t)
        print(f"[Round {round_t}] Selected Clients via UCB: {self.online_clients}")


        for i in self.online_clients:
            # self._train_net(i, self.nets_list[i], priloader_list[i])
            # train_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            train_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            reward = -train_loss  # lower loss = better client
            self.q_score[i] = reward
            self.client_stats[i]["total_reward"] += reward
            self.client_stats[i]["num_selections"] += 1



            if self.args.wHEAL == 1:
                net_params = self.nets_list[i].state_dict()
                global_params = self.global_net.state_dict()
                param_names = [name for name, _ in self.nets_list[i].named_parameters()]
                update_diff = {key: global_params[key] - net_params[key] for key in global_params}
                
                
                #  Ensure history dicts are initialized before consistency_mask
                if i not in self.increase_history:
                    self.increase_history[i] = {}
                if hasattr(self, 'decrease_history') and i not in self.decrease_history:
                    self.decrease_history[i] = {}

                mask = self.consistency_mask(i, update_diff)
                self.mask_dict[i] = mask
                masked_update = {key: update_diff[key] * mask[key] for key in update_diff} 
                self.client_update[i] = masked_update
                    
                self.compute_distance(i, self.client_update[i], param_names)

        freq = self.get_params_diff_weights()
        self.aggregate_nets_parameter(freq)

        #  Update rewards for selected clients
        
        # for m in self.online_clients:
        #     reward = self.compute_reward(m)
        #     self.q_score[m] = reward
        #     self.client_stats[m]["total_reward"] += reward
        #     self.client_stats[m]["num_selections"] += 1




    
    # def _train_net(self, index, net, train_loader):
    #     net = net.to(self.device)

    #     # if val_loader is not None:
    #     #     self.val_acc_before[index] = self.evaluate(net, val_loader)

    #     net.train()
    #     optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
    #     criterion = nn.CrossEntropyLoss()
    #     criterion.to(self.device)
    #     iterator = tqdm(range(self.local_epoch))

    #     for _ in iterator:
    #         for batch_idx, (images, labels) in enumerate(train_loader):
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)
    #             outputs = net(images)
    #             loss = criterion(outputs, labels)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
    #             optimizer.step()

    #     # if val_loader is not None:
    #     #     self.val_acc_after[index] = self.evaluate(net, val_loader)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.local_epoch)

        criterion = nn.CrossEntropyLoss().to(self.device)
        iterator = tqdm(range(self.local_epoch))

        total_loss = 0.0
        total_samples = 0

        for local_ep in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

            scheduler.step()
            avg_loss = total_loss / total_samples
            iterator.desc = f"Local Participant {index} loss = {avg_loss:.3f}"

        return avg_loss
        
            



