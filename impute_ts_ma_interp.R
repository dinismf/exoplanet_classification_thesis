library(imputeTS)

#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original.csv", header = FALSE)
#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original_normed.csv", header = FALSE)
data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original_std.csv", header = FALSE)


data_nan_interpolated = na.interpolation(data)
data_nan_movingaverage = na.ma(data) 


data_nan_ma_single = data_nan_movingaverage[sample(nrow(data_nan_movingaverage), 1), ]
data_nan_interpolated_single = data_nan_interpolated[sample(nrow(data_nan_interpolated), 1), ]


row = rownames(data_nan_ma_single)
row2 = rownames(data_nan_interpolated_single)


data_single = data[row,]
data2_single = data[row2,]

data_single = as.ts(data_single)
data2_single = as.ts(data2_single)
data_single = as.vector(data_single)
data2_single = as.vector(data2_single)

data_nan_ma_single = as.ts(data_nan_ma_single)
data_nan_interpolated_single = as.ts(data_nan_interpolated_single)
data_nan_ma_single = as.vector(data_nan_ma_single)
data_nan_interpolated_single = as.vector(data_nan_interpolated_single)

plotNA.imputations(x.withNA = data_single, x.withImputations = data_nan_ma_single)
plotNA.imputations(x.withNA = data2_single, x.withImputations = data_nan_interpolated_single)

plotNA.distribution(data_single)

write.csv(data_nan_interpolated, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_interpolated_nan\\interpolated_nan_data.csv")
write.csv(data_nan_movingaverage, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_movingaverage_nan\\movingaverage_nan_data.csv")

help("read.csv")
