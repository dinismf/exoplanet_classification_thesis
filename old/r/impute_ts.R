library(imputeTS)

#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original.csv", header = FALSE)
#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original_normed.csv", header = FALSE)
data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original_std.csv", header = FALSE)


data_nan_kalman = na.kalman(data,model = 'auto.arima')
data_nan_seadec = na.seadec(data)


data_nan_kalman_single = data_nan_kalman[sample(nrow(data_nan_kalman), 1), ]
data_nan_seadec_single = data_nan_seadec[sample(nrow(data_nan_seadec), 1), ]


#kalman_dim = dim(data_nan_kalman_single)
  
  
row = rownames(data_nan_kalman_single)
row2 = rownames(data_nan_ma_single)
row3 = rownames(data_nan_interpolated_single)
row4 = rownames(data_nan_seadec_single)


data_single = data[row,]
data2_single = data[row2,]
data3_single = data[row3,]
data4_single = data[row4,]

data_single = as.ts(data_single)
data2_single = as.ts(data2_single)
data3_single = as.ts(data3_single)
data4_single = as.ts(data4_single)
data_single = as.vector(data_single)
data2_single = as.vector(data2_single)
data3_single = as.vector(data3_single)
data4_single = as.vector(data4_single)


data_dim = dim(data_single)


data_nan_kalman_single = as.ts(data_nan_kalman_single)
data_nan_ma_single = as.ts(data_nan_ma_single)
data_nan_interpolated_single = as.ts(data_nan_interpolated_single)
data_nan_seadec_single = as.ts(data_nan_seadec_single)
data_nan_kalman_single = as.vector(data_nan_kalman_single)
data_nan_ma_single = as.vector(data_nan_ma_single)
data_nan_interpolated_single = as.vector(data_nan_interpolated_single)
data_nan_seadec_single = as.vector(data_nan_seadec_single)

plotNA.imputations(x.withNA = data2_single, x.withImputations = data_nan_ma_single)
plotNA.imputations(x.withNA = data3_single, x.withImputations = data_nan_interpolated_single)

plotNA.distribution(data_single)

write.csv(data_nan_interpolated, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_interpolated_nan\\interpolated_nan_data.csv")
write.csv(data_nan_kalman, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_kalman_nan\\kalman_nan_data.csv")
write.csv(data_nan_movingaverage, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_movingaverage_nan\\movingaverage_nan_data.csv")

help("read.csv")
