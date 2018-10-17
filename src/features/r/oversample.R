library(OSTSC)

data <- read.csv("C:\\Users\\DYN\\Desktop\\exoplanet_classification_repo\\data\\lc_std_nanimputed.csv")

y <- as.array(data[,c("LABEL")])


labelcol <- names(data) %in% c("LABEL") 

table(y)

X <- as.matrix(data[!labelcol])

is.recursive(X)
is.recursive(y)
is.atomic(X)
is.atomic(y)

dim(X)
dim(y)

NewData <- OSTSC(X,y, parallel = FALSE)

NewX <- NewData$sample
NewY <- NewData$label

table(NewY)