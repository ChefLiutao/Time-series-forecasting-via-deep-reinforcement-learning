# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:20:16 2020

@author: ChefLiutao
"""

class Worker():
    def __init__(self, name, globalAC):
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state_index = np.random.choice(range(len(train_state_mat)))
            s = train_state_mat[state_index]
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                
                s_ = train_state_mat[state_index+1]
                r = -abs(train_best_action[state_index]-a)
                if ep_t == MAX_EP_STEP -1 or state_index == len(train_state_mat)-2:
                    done = True
                else:
                    done = False
                
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
            #    buffer_r.append((r+8)/8)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()   #把中央大脑最新的参数拉下来

                s = s_
                state_index += 1
                total_step += 1
                
                #完成一个episode，查看训练效果
#                if done:
#                    self.AC.pull_global()
                    
                    #训练集
#                    pred = []
#                    for i_pred in range(len(train_state_mat)):
#                        state = train_state_mat[i_pred]
#                        action = self.AC.choose_action(state)
#                        pred.append(action)
#                    pred = [pred[i][0] for i in range(len(train_state_mat))]
#                    pred = pd.Series(pred)
#                    pred = pred*(B-A)+A
#                    actual = train_best_action*(B-A)+A   #反归一化
#                    MAE = np.mean(abs(pred-actual)) #MAE
#                    RMSE = (np.sum((pred-actual)**2)/len(pred))**0.5
#                    print('MAE-1: %.2f' %MAE,'RMSE-1: %.2f' %RMSE)
#                    mae_episode.append(MAE)
#                    rmse_episode.append(RMSE)

                    #测试集
#                    pred = []
#                    for i_pred in range(len(test_state_mat)):
#                        state = test_state_mat[i_pred]
#                        action = self.AC.choose_action(state)
#                        pred.append(action)
#                    pred = [pred[i][0] for i in range(len(test_state_mat))]
#                    pred = pd.Series(pred)
#                    pred = pred*(B-A)+A
#                    actual = test_best_action*(B-A)+A   #反归一化
#                    MAE = np.mean(abs(pred-actual)) #MAE
#                    RMSE = (np.sum((pred-actual)**2)/len(pred))**0.5
#                    print('MAE: %.2f' %MAE,'RMSE: %.2f' %RMSE)
               
                
                #完成一个episode
                if done:  
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break