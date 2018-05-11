library(forecast)

data = read.csv("C:\\Users\\DYN\\Google Drive\\Intelligent_Systems_MSc\\MSc_Project\\data\\main\\original_lc\\planets_labelled_final_original.csv", header = FALSE)

data_single = data[sample(nrow(data_nan_movingaverage), 1), ]


data_single = data[row,]
data_single = as.ts(data_single)
data_single = as.vector(data_single)
row = rownames(data_nan_ma_single)



missindx <- is.na(data)

arimaModel <- auto.arima(data)
model <- arimaModel$model

#Kalman smoothing
kal <- KalmanSmooth(data, model, nit )
erg <- kal$smooth  

for ( i in 1:length(model$Z)) {
  erg[,i] = erg[,i] * model$Z[i]
}
karima <-rowSums(erg)

for (i in 1:length(data)) {
  if (is.na(data[i])) {
    data[i] <- karima[i]
  }
}

#Original TimeSeries with imputed values
print(data)

write.csv(data_nan_interpolated, "interpolated_nan_data.csv")
write.csv(data_nan_kalman, "kalman_nan_data.csv")
write.csv(data_nan_movingaverage, "movingaverage_nan_data.csv")
write.csv(data, "kalman_nan_data.csv")

help("read.csv")
