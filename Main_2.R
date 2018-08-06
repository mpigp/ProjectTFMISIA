####################
#Carga de la librerias necesarias
####################

library("keras")
library("tensorflow")
library("reticulate")
library(data.table)

K <- backend()

library(rminer) 
library(caret)
library(pROC)
library(FSelector)
library(nnet)
library(e1071)
library(rpart)
library(RWeka)

# **********************
# Fase no supervisada 
# **********************

# ########################################
# CARGA Y PREPARACION DE DATOS DE DATOS
# ########################################

# leer fichero  con el conjunto de datos
datos_test <- read.arff("breastCancer-test.arff")
datos_train <- read.arff("breastCancer-train.arff")

# Preparacion de datos para el entrenamiento y validación del autoencoder
x_train <- subset.data.frame(datos_train,select = -Class)
x_test <- subset.data.frame(datos_test,select = -Class)

# Convertirlo en tipo matriz
x_train_matrix = data.matrix(x_train)
x_test_matrix = data.matrix(x_test)

print(cat('Conjunto de entrenamiento. Número de columna: ',ncol(x_train_matrix),' Número de filas: ',nrow(x_train_matrix)))
print(cat('Conjunto de test Número de columna: ',ncol(x_test_matrix),' Número de filas: ',nrow(x_test_matrix)))

# ####################
#    AUTOENCODER
# ####################

# ------------------------
# Variables de inicio
# ------------------------

# Numero de neuronas de entrada
original_dim <- ncol(x_train_matrix)
# Numero de neuronas de la capa oculta 1 y 2
encoding_dim_1 <- round(ncol(x_train_matrix)/2 )
encoding_dim_2 <- round(encoding_dim_1/2)
encoding_dim_3 <- round(encoding_dim_2/2)
encoding_dim_4 <- round(encoding_dim_3/2)
encoding_dim_5 <- round(encoding_dim_4/2)
encoding_dim_6 <- round(encoding_dim_5/2)
encoding_dim_7 <- round(encoding_dim_6/2)
encoding_dim_8 <- round(encoding_dim_7/2)
encoding_dim_9 <- round(encoding_dim_8/2)
encoding_dim_10 <- round(encoding_dim_9/2)
encoding_dim_11 <- 8
# Número de neurona de salida
latent_dim <- ncol(x_train_matrix)

# ------------------------
# Definicion del modelo 
# ------------------------

# Capa de entrada del autoencoder
input <- layer_input(shape = c(original_dim))

# La capa de codificación, donde se le pasa la capa de entrada, dimesion (numero de neuronas), funcion de activacion
encoded_1<- layer_dense(input,encoding_dim_1 , activation = "relu")
encoded_2<- layer_dense(input,encoding_dim_2 , activation = "relu")
encoded_3<- layer_dense(input,encoding_dim_3 , activation = "relu")
encoded_4<- layer_dense(input,encoding_dim_4 , activation = "relu")
encoded_5<- layer_dense(input,encoding_dim_5 , activation = "relu")
encoded_6<- layer_dense(input,encoding_dim_6 , activation = "relu")
encoded_7<- layer_dense(input,encoding_dim_7 , activation = "relu")
encoded_8<- layer_dense(input,encoding_dim_8 , activation = "relu")
encoded_9<- layer_dense(input,encoding_dim_9 , activation = "relu")
encoded_10<- layer_dense(input,encoding_dim_10 , activation = "relu")
encoded_11<- layer_dense(input,encoding_dim_11 , activation = "relu")

# La capa de decodificacion, vuelve a tener las dimensiones de la entrada existe una perdida de informacion con respecto a la entrada
# Como prámetros se utiliza la salida de la capa de codificacion,tamaño de la capa de decodificacion (latent_dim), funcion de activacion (sigmoid) 
decoded<- layer_dense(encoded_11, latent_dim , activation = "sigmoid")
# si usamos sigmoid la salida comprende los valores entre 0 y 1

# -------------------------------------
# Generamos los modelos
# -------------------------------------

# modelo de codificacion
model_enconded <- keras_model(input, encoded_11)

# modelo de autoencoder
model_autoencoder <- keras_model(input, decoded)

# --------------------------
# Compilacion del modelo 
# --------------------------

model_autoencoder  %>% compile(optimizer = 'adam', loss = 'binary_crossentropy')

summary(model_autoencoder)

# --------------------------
# Entrenamiento del modelo 
# --------------------------

history <- model_autoencoder %>% fit(x_train_matrix, x_train_matrix,epochs=50,batch_size=256)

# Pintar resultado del entrenamiento
plot(history)

# --------------------------
# Prediccion de los modelos 
# --------------------------
feature_1 <- predict(object = model_enconded,x = x_test_matrix)
feature_2 <- predict(object = model_autoencoder,x = x_test_matrix)

# Guardar datos en ficheros
write.csv(feature_1, file = "data_encoded.csv",row.names = TRUE)
write.csv(feature_2, file = "data_autoencoded.csv",row.names = TRUE)

View(feature_1)

# **************************
# Fase supervisada
# **************************

#recopilacion <- data.frame("Predictorio","Test","ACC","Metodo")
# ##############################################
# Realizacion de la funcion de cross validation
# ##############################################
crossVal <- function(x,tipo,tamK= 10,n=5,nnetTam=5,metodo,svmGam=1,svmC=1 ){
  
  #Creación de la partición del conjunto de datos de entrada
  folds <- createFolds(x$Class,k = tamK)
  
  # Inicializar variables locales
  ACC_media <- 0
  
  # Comienza el calculo del ACC del modelo predictivo
  for (i in 1:tamK){
    # Se obtiene el conjunto de test y entrenamiendo por vuelta en el bucle
    test <- x[folds[[i]],]
    train <- x[-folds[[i]],]
    
    #Dependiento del método se va a realizar una función u otra, se controla a traves de un switch
    switch(metodo,
           nnet={
             #Varía la variable "size" dependiento de lo que se pasa como entrada "tam"
             print("Se ejecuta la función de nnet")
             # Creacion del modelo predictivo y la prediccion
             mod <- nnet(Class~.,data=train,size=nnetTam,maxit=1000,decay=5e-4)
             pre <- predict(mod,test,type = "class")
           },
           e1071={
             #Varía la variable "cost" y "gamma" dependiento de lo que se pasa como entrada "C" y"gam"
             print("Se ejecuta la función de svm")
             # Creacion del modelo predictivo y la prediccion
             mod <- svm(Class~.,data=train,cost=svmC,gamma=svmGam)
             pre <- predict(mod,test,probability=TRUE)
           },
           rpart={
             print("Se ejecuta la función de rpart")
             mod <- rpart(Class~.,train)
             pre_prob <- predict(mod,test,type="prob")
             aux_tree <- data.frame(pre_prob)
             final <- test$Class
           },
           stop("Método introducido no correcto")
    )
    
    #Una vez realizada la predicción de calcula el ACC
    
    if(metodo=="rpart"){
      for(j in 1:length(aux_tree$SI)){
        if (aux_tree$SI[j] >= 0.5){
          final[j] = "SI"
        }else{
          final[j] = "NO"
        }
      }
      pre <- final
    }
    
    #Acumular el valor ACC obtenido
    #aux <- ACC(pre,test$recid)
    aux <- sum(pre == test$Class)/length(pre)
    #rbind(recopilacion,list(pre,test$Class,aux,metodo))
    ACC_media <- ACC_media + aux
  }
  
  #Devolvemos el valor medio de AUC
  return(ACC_media/length(folds))
}

# ########################################
# CARGA Y PREPARACION DE DATOS DE DATOS
# ########################################

DatosCodificados <- read.csv(file="data_encoded.csv",header = TRUE)
DatosCodificados[paste("Class")] <- datos_test$Class
DatosCodificados$Class <- ifelse(DatosCodificados$Class=="non-relapse","NO","SI")
DatosCodificados$Class <- as.factor(DatosCodificados$Class)
DatosCodificados <- subset.data.frame(DatosCodificados,select = -X)

ACC_rd_3 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 3)
ACC_rd_5 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 5)
ACC_rd_7 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 7)
ACC_rd_10 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 10)
ACC_rd_13 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 13)
ACC_rd_15 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 15)
ACC_rd_17 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 17)
ACC_rd_20 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 20)
ACC_rd_30 <- crossVal(DatosCodificados,tamK = 5,metodo = "nnet",nnetTam = 30)

 #Se aglutinan los datos utilizados y resultados para representarlos en una gráfica
size <- c(3,5,7,10,13,15,17,20,30)
ACC <- c(ACC_rd_3,ACC_rd_5,ACC_rd_7,ACC_rd_10,ACC_rd_13,ACC_rd_15,ACC_rd_17,ACC_rd_20,ACC_rd_30)
tabla_rn <- data.frame(ACC,size)
plot(tabla_rn$size,tabla_rn$ACC,main = "Comportamiento de nnet",xlab =  "Size",ylab ="ACC medio")
tabla_rn
# 
# 
# 
# 
# #Solución para: Support Vector Machines
# ACC_vm_1 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=1,svmC=1)
# ACC_vm_2 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=1,svmC=50)
# ACC_vm_3 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=1,svmC=60)
# ACC_vm_4 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=1,svmC=80)
# ACC_vm_5 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=1,svmC=90)
# ACC_vm_6 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=50,svmC=1)
# ACC_vm_7 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=60,svmC=50)
# ACC_vm_8 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=80,svmC=60)
# ACC_vm_9 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=90,svmC=80)
# ACC_vm_10 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=100,svmC=90)
# ACC_vm_11 <- crossVal(DatosCodificados,tamK = 5,metodo="e1071",svmGam=1000,svmC=100)
# 
# #Se aglutinan los datos utilizados y resultados para representarlos en una gráfica
# ACC= c(ACC_vm_1,ACC_vm_2,ACC_vm_3,ACC_vm_4,ACC_vm_5,ACC_vm_6,ACC_vm_7,ACC_vm_8,ACC_vm_9,ACC_vm_10,ACC_vm_11)
# gamma= c(1,1,1,1,1,50,60,80,90,100,1000)
# cost= c(1,50,60,80,90,1,50,60,80,90,100)
# tabla_vm <- data.frame(ACC,gamma,cost)
# #install.packages("scatterplot3d")
# require("scatterplot3d")
# scatterplot3d(tabla_vm,main = "Comportamiento función svm",type = "h",box = FALSE,angle = 60)
# tabla_vm
# 
# 
