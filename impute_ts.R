library(imputeTS)

data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original.csv", header = FALSE)
data[1] <- NULL

data_nan_interpolated = na.interpolation(data)
data_nan_kalman = na.kalman(data,model = 'auto.arima')
#data_nan_seadec = na.seadec(data)
#data_nan_movingaverage = na.ma(data, k = 4, weighting = "simple") 

#a = tsAirgap

#b = tsAirgapComplete

#c = na.kalman(a)


data_nan_kalman_single = data_nan_kalman[sample(nrow(data_nan_kalman), 1), ]
#data_nan_ma_single = data_nan_movingaverage[sample(nrow(data_nan_movingaverage), 1), ]


#kalman_dim = dim(data_nan_kalman_single)
  
  
row = rownames(data_nan_kalman_single)
#row = rownames(data_nan_ma_single)

data_single = data[row,]
data_single = as.ts(data_single)
data_single = as.vector(data_single)

data_dim = dim(data_single)


data_nan_kalman_single = as.ts(data_nan_kalman_single)
data_nan_kalman_single = as.vector(data_nan_kalman_single)

#ata_nan_ma_single = as.ts(data_nan_ma_single)
#data_nan_ma_single = as.vector(data_nan_ma_single)


plotNA.imputations(x.withNA = data_single, x.withImputations = data_nan_kalman_single)
plotNA.distribution(data_single)

write.csv(data_nan_interpolated, "interpolated_nan_data.csv")
write.csv(data_nan_kalman, "kalman_nan_data.csv")
write.csv(data_nan_movingaverage, "movingaverage_nan_data.csv")

help("read.csv")
