#coding:utf-8

#pv,click,score

@annotate('*->double')
class GetAuc:
    def AUC(self,buffer):  #buffer is a list ,element format is [int(round(float(pv))),int(round(float(click))),float(score)]
        pv_num,click_num,_ = [sum(x) for x in zip(*buffer)]
        non_click_num = pv_num - click_num
        if (click_num ==0 or non_click_num == 0): 
            return 0.5 
        pre_score = 0.0 
        pre_rank = 0 
        rank = 0 
        sum_rank = 0.0 
        positive_sample_num = 0 
        equal_sample_num = 0 
        for index,pair in enumerate(sorted(buffer, key=lambda x:x[2], reverse = False)):
            pv,click,score = pair
            #print rank
            if rank == 0:
                rank = pv  
                pre_rank = 1 
                pre_score = score
                if click > 0:
                    equal_sample_num += click
                continue
            if abs(pre_score-score) < 1e-9:
                rank += pv
                if click > 0:
                    equal_sample_num += click
                continue
            
            sum_rank += equal_sample_num * (pre_rank + rank) / 2.0 
            if click > 0:
                equal_sample_num = click 
            else:
                equal_sample_num = 0
            pre_rank = rank + 1
            rank += pv
            pre_score = score
        sum_rank += equal_sample_num * (pre_rank + rank) / 2.0
        auc = (sum_rank - (click_num * (click_num + 1)) / 2) / (click_num * non_click_num)
        return auc
        