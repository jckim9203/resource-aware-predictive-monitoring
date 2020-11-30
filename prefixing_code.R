setwd('/Resource/data_with_exp/data_with_exp')
data_bpic11_f1 <- read.csv('BPIC11_f1_exp.csv',header=T, sep=";")
data_bpic11_f2 <- read.csv('BPIC11_f2_exp.csv',header=T, sep=";")
data_bpic11_f3 <- read.csv('BPIC11_f3_exp.csv',header=T, sep=";")
data_bpic11_f4 <- read.csv('BPIC11_f4_exp.csv',header=T, sep=";")
data_bpic15_1_f2 <- read.csv('BPIC15_1_f2_exp.csv',header=T, sep=";")
data_bpic15_2_f2 <- read.csv('BPIC15_2_f2_exp.csv',header=T, sep=";")
data_bpic15_3_f2 <- read.csv('BPIC15_3_f2_exp.csv',header=T, sep=";")
data_bpic15_4_f2 <- read.csv('BPIC15_4_f2_exp.csv',header=T, sep=";")
data_bpic15_5_f2 <- read.csv('BPIC15_5_f2_exp.csv',header=T, sep=";")
data_bpic17_accepted <- read.csv('BPIC17_O_Accepted_exp.csv',header=T, sep=";")
data_bpic17_cancelled <- read.csv('BPIC17_O_Cancelled_exp.csv',header=T, sep=";")
data_bpic17_refused <- read.csv('BPIC17_O_Refused_exp.csv',header=T, sep=";")
data_bpic11_f1_sorted <- data_bpic11_f1[order(data_bpic11_f1$Case.ID),]
data_bpic11_f2_sorted <- data_bpic11_f2[order(data_bpic11_f2$Case.ID),]
data_bpic11_f3_sorted <- data_bpic11_f3[order(data_bpic11_f3$Case.ID),]
data_bpic11_f4_sorted <- data_bpic11_f4[order(data_bpic11_f4$Case.ID),]
data_bpic15_1_f2_sorted <- data_bpic15_1_f2[order(data_bpic15_1_f2$Case.ID),]
data_bpic15_2_f2_sorted <- data_bpic15_2_f2[order(data_bpic15_2_f2$Case.ID),]
data_bpic15_3_f2_sorted <- data_bpic15_3_f2[order(data_bpic15_3_f2$Case.ID),]
data_bpic15_4_f2_sorted <- data_bpic15_4_f2[order(data_bpic15_4_f2$Case.ID),]
data_bpic15_5_f2_sorted <- data_bpic15_5_f2[order(data_bpic15_5_f2$Case.ID),]
data_bpic17_accepted_sorted <- data_bpic17_accepted[order(data_bpic17_accepted$Case.ID),]
data_bpic17_cancelled_sorted <- data_bpic17_cancelled[order(data_bpic17_cancelled$Case.ID),]
data_bpic17_refused_sorted <- data_bpic17_refused[order(data_bpic17_refused$Case.ID),]
data_bpic11_f1['prefix'] = 1
data_bpic11_f2['prefix'] = 1
data_bpic11_f3['prefix'] = 1
data_bpic11_f4['prefix'] = 1
data_bpic15_1_f2['prefix'] = 1
data_bpic15_2_f2['prefix'] = 1
data_bpic15_3_f2['prefix'] = 1
data_bpic15_4_f2['prefix'] = 1
data_bpic15_5_f2['prefix'] = 1
data_bpic17_accepted['prefix'] = 1
data_bpic17_cancelled['prefix'] = 1
data_bpic17_refused['prefix'] = 1
data_bpic11_f1_sorted['prefix'] = 1
data_bpic11_f2_sorted['prefix'] = 1
data_bpic11_f3_sorted['prefix'] = 1
data_bpic11_f4_sorted['prefix'] = 1
data_bpic15_1_f2_sorted['prefix'] = 1
data_bpic15_2_f2_sorted['prefix'] = 1
data_bpic15_3_f2_sorted['prefix'] = 1
data_bpic15_4_f2_sorted['prefix'] = 1
data_bpic15_5_f2_sorted['prefix'] = 1
data_bpic17_accepted['prefix'] = 1
data_bpic17_cancelled['prefix'] = 1
data_bpic17_refused['prefix'] = 1
for(i in 1:nrow(data_bpic11_f1_sorted)){
  if(i == nrow(data_bpic11_f1_sorted)){
    break;
  }
  if(data_bpic11_f1_sorted[i,3] == data_bpic11_f1_sorted[i+1,3]){
    data_bpic11_f1_sorted$prefix[i+1] = data_bpic11_f1_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic11_f2_sorted)){
  if(i == nrow(data_bpic11_f2_sorted)){
    break;
  }
  if(data_bpic11_f2_sorted[i,3] == data_bpic11_f2_sorted[i+1,3]){
    data_bpic11_f2_sorted$prefix[i+1] = data_bpic11_f2_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic11_f3_sorted)){
  if(i == nrow(data_bpic11_f3_sorted)){
    break;
  }
  if(data_bpic11_f3_sorted[i,3] == data_bpic11_f3_sorted[i+1,3]){
    data_bpic11_f3_sorted$prefix[i+1] = data_bpic11_f3_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic11_f4_sorted)){
  if(i == nrow(data_bpic11_f4_sorted)){
    break;
  }
  if(data_bpic11_f4_sorted[i,3] == data_bpic11_f4_sorted[i+1,3]){
    data_bpic11_f4_sorted$prefix[i+1] = data_bpic11_f4_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic15_1_f2_sorted)){
  if(i == nrow(data_bpic15_1_f2_sorted)){
    break;
  }
  if(data_bpic15_1_f2_sorted[i,3] == data_bpic15_1_f2_sorted[i+1,3]){
    data_bpic15_1_f2_sorted$prefix[i+1] = data_bpic15_1_f2_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic15_2_f2_sorted)){
  if(i == nrow(data_bpic15_2_f2_sorted)){
    break;
  }
  if(data_bpic15_2_f2_sorted[i,3] == data_bpic15_2_f2_sorted[i+1,3]){
    data_bpic15_2_f2_sorted$prefix[i+1] = data_bpic15_2_f2_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic15_3_f2_sorted)){
  if(i == nrow(data_bpic15_3_f2_sorted)){
    break;
  }
  if(data_bpic15_3_f2_sorted[i,3] == data_bpic15_3_f2_sorted[i+1,3]){
    data_bpic15_3_f2_sorted$prefix[i+1] = data_bpic15_3_f2_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic15_4_f2_sorted)){
  if(i == nrow(data_bpic15_4_f2_sorted)){
    break;
  }
  if(data_bpic15_4_f2_sorted[i,3] == data_bpic15_4_f2_sorted[i+1,3]){
    data_bpic15_4_f2_sorted$prefix[i+1] = data_bpic15_4_f2_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic15_5_f2_sorted)){
  if(i == nrow(data_bpic15_5_f2_sorted)){
    break;
  }
  if(data_bpic15_5_f2_sorted[i,3] == data_bpic15_5_f2_sorted[i+1,3]){
    data_bpic15_5_f2_sorted$prefix[i+1] = data_bpic15_5_f2_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic17_accepted_sorted)){
  if(i == nrow(data_bpic17_accepted_sorted)){
    break;
  }
  if(data_bpic17_accepted_sorted[i,3] == data_bpic17_accepted_sorted[i+1,3]){
    data_bpic17_accepted_sorted$prefix[i+1] = data_bpic17_accepted_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic17_cancelled_sorted)){
  if(i == nrow(data_bpic17_cancelled_sorted)){
    break;
  }
  if(data_bpic17_cancelled_sorted[i,3] == data_bpic17_cancelled_sorted[i+1,3]){
    data_bpic17_cancelled_sorted$prefix[i+1] = data_bpic17_cancelled_sorted$prefix[i] + 1
  } 
}
for(i in 1:nrow(data_bpic17_refused_sorted)){
  if(i == nrow(data_bpic17_refused_sorted)){
    break;
  }
  if(data_bpic17_refused_sorted[i,3] == data_bpic17_refused_sorted[i+1,3]){
    data_bpic17_refused_sorted$prefix[i+1] = data_bpic17_refused_sorted$prefix[i] + 1
  } 
}
write.csv(data_bpic11_f1_sorted, "BPIC11_f1_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic11_f2_sorted, "BPIC11_f2_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic11_f3_sorted, "BPIC11_f3_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic11_f4_sorted, "BPIC11_f4_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic15_1_f2_sorted, "BPIC15_1_f2_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic15_2_f2_sorted, "BPIC15_2_f2_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic15_3_f2_sorted, "BPIC15_3_f2_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic15_4_f2_sorted, "BPIC15_4_f2_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic15_5_f2_sorted, "BPIC15_5_f2_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic17_accepted_sorted, "BPIC17_accepted_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic17_cancelled_sorted, "BPIC17_cancelled_exp_prefix.csv", row.names=FALSE)
write.csv(data_bpic17_refused_sorted, "BPIC17_refused_exp_prefix.csv", row.names=FALSE)
