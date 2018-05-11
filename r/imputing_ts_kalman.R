library(imputeTS)

#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original.csv", header = FALSE)
#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original_normed.csv", header = FALSE)
#data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original_std.csv", header = FALSE)
data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_smoothed_std.csv", header = FALSE)

# Remove missing values from the original dataset with several different imputation methods
data_nan_kalman_default = na.kalman(data)
data_nan_kalman_arima = na.kalman(data,model = 'auto.arima')

# Sample random row from original data
data_single = data[sample(nrow(data_nan_kalman_default), 1),]

# Obtain row number 
row = rownames(data_single)

# Convert sampled row to vector 
data_single = as.ts(data_single)
data_single = as.vector(data_single)


# Sample row number from imputed datasets 
data_nan_kalman_default_single = data_nan_kalman_default[row,]
data_nan_kalman_arima_single = data_nan_kalman_arima[row, ]


data_nan_kalman_default_single = as.ts(data_nan_kalman_default_single)
data_nan_kalman_default_single = as.vector(data_nan_kalman_default_single)
data_nan_kalman_arima_single = as.ts(data_nan_kalman_arima_single)
data_nan_kalman_arima_single = as.vector(data_nan_kalman_arima_single)

plotNA.imputations(x.withNA = data_single, x.withImputations = data_nan_kalman_default_single)
plotNA.imputations(x.withNA = data_single, x.withImputations = data_nan_kalman_arima_single)

plotNA.distribution(data_single)

write.csv(data_nan_kalman_default, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_kalman_nan\\kalman_default_nan_data.csv")
write.csv(data_nan_kalman_arima, "C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\lc_kalman_nan\\kalman_arima_nan_data.csv")

help("read.csv")
