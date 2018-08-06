####################
#Carga de la librerias necesarias
####################

library("rminer")
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

# **********************
# Fase no supervisada 
# **********************

# ########################################
# CARGA Y PREPARACION DE DATOS DE DATOS
# ########################################

# leer fichero  con el conjunto de datos
Datos <- read.csv(file="breast-cancer.csv",header = TRUE)


# Edad: 20-29 (1),30-39 (2), 40-49 (3), 50-59 (4), 60-69 (5), 70-79 (6)
# convertir a datos categoricos
# for(unique_value in unique(Datos$age)){
#   Datos[paste("age", unique_value, sep = ".")] <- ifelse(Datos$age == unique_value, 1, 0)
# }
# Datos <- subset.data.frame(Datos,select = -age)

Datos$age <- as.numeric(Datos$age)
for (k in length(Datos$age)){
  if (Datos$age[k]=="20-29"){
    Datos$age[k] <- 1
  } else if(Datos$age[k]=="30-39"){
    Datos$age[k] <- 2
  } else if(Datos$age[k]=="40-49"){
    Datos$age[k] <- 3
  } else if(Datos$age[k]=="50-59"){
    Datos$age[k] <- 4
  } else if(Datos$age[k]=="60-69"){
    Datos$age[k] <- 5
  } else if(Datos$age[k]=="70-79"){
    Datos$age[k] <- 6
  }
}

# mefalsepause: ge40 (1), lt40 (2), premefalse (3)

# for(unique_value in unique(Datos$mefalsepause)){
#   Datos[paste("mefalsepause", unique_value, sep = ".")] <- ifelse(Datos$mefalsepause == unique_value, 1, 0)
# }
# 
# Datos <- subset.data.frame(Datos,select = -mefalsepause)

Datos$mefalsepause <- as.numeric(Datos$mefalsepause)
for (k in length(Datos$mefalsepause)){
  if (Datos$mefalsepause[k]=="ge40"){
    Datos$mefalsepause[k] <- 1
  } else if(Datos$mefalsepause[k]=="lt40"){
    Datos$mefalsepause[k] <- 2
  } else if(Datos$mefalsepause[k]=="premefalse"){
    Datos$mefalsepause[k] <- 3
  }
}

# 0-4 (1), 10-14 (2), 15-19 (3), 20-24 (4), 25-29 (6), 30-34 (7), 35-39 (8), 40-44 (9), 45-49 (10), 5-9 (11), 50-54 (12)

for(unique_value in unique(Datos$tumor.size)){
  Datos[paste("tumor.size", unique_value, sep = ".")] <- ifelse(Datos$tumor.size == unique_value, 1, 0)
}

Datos <- subset.data.frame(Datos,select = -tumor.size)

for(unique_value in unique(Datos$inv.falsedes)){
  Datos[paste("inv.falsedes", unique_value, sep = ".")] <- ifelse(Datos$inv.falsedes == unique_value, 1, 0)
}

Datos <- subset.data.frame(Datos,select = -inv.falsedes)

for(unique_value in unique(Datos$breast.quad)){
  Datos[paste("breast.quad", unique_value, sep = ".")] <- ifelse(Datos$breast.quad == unique_value, 1, 0)
}

Datos <- subset.data.frame(Datos,select = -breast.quad)


Datos$falsede.caps <- ifelse(Datos$falsede.caps=="true",1,0)
Datos$breast <- ifelse(Datos$breast=="right",1,0)
Datos$irradiat <- ifelse(Datos$irradiat=="true",1,0)



# Información del conjunto de datos
summary(Datos)

# Metodo de validacion interna holdout, para el uso del autoencoder 
# Creacion de subconjutnos de entrenamiento (334) y test (166)
Division <- holdout(y=Datos$class)

# Preparacion de datos para el entrenamiento y validación del autoencoder
train <- Datos[Division$tr,]
test <- Datos[Division$ts,]
x_train <- subset.data.frame(train,select = -class)
x_test <- subset.data.frame(test,select = -class)

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
encoding_dim_1 <- round(ncol(x_train_matrix)/2)
encoding_dim_2 <- 7
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

# La capa de decodificacion, vuelve a tener las dimensiones de la entrada existe una perdida de informacion con respecto a la entrada
# Como prámetros se utiliza la salida de la capa de codificacion,tamaño de la capa de decodificacion (latent_dim), funcion de activacion (sigmoid) 
decoded<- layer_dense(encoded_2, latent_dim , activation = "sigmoid")
# si usamos sigmoid la salida comprende los valores entre 0 y 1

# -------------------------------------
# Generamos los modelos
# -------------------------------------

# modelo de codificacion
model_enconded <- keras_model(input, encoded_2)

# modelo de autoencoder
model_autoencoder <- keras_model(input, decoded)

# --------------------------
# Compilacion del modelo 
# --------------------------

model_autoencoder  %>% compile(optimizer = 'adam', loss = 'binary_crossentropy')

summary(model_autoencoder)
summary(model_enconded)

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
View(feature_2)


# **************************
# Fase supervisada
# **************************


# ##############################################
# Realizacion de la funcion de cross validation
# ##############################################
crossVal <- function(x,tipo,tamK= 10,n=5,nnetTam=5,metodo,svmGam=1,svmC=1 ){
  
  #Creación de la partición del conjunto de datos de entrada
  folds <- createFolds(x$class,k = tamK)
  
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
             # Artificial neural networks 
             #Varía la variable "size" dependiento de lo que se pasa como entrada "tam"
             print("Se ejecuta la función de nnet")
             # Creacion del modelo predictivo y la prediccion
             mod <- nnet(class~.,data=train,size=nnetTam,maxit=1000,decay=5e-4)
             pre <- predict(mod,test,type = "class")
           },
           e1071={
             # Máquinas de soporte vectorial
             #Varía la variable "cost" y "gamma" dependiento de lo que se pasa como entrada "C" y"gam"
             print("Se ejecuta la función de svm")
             # Creacion del modelo predictivo y la prediccion
             mod <- svm(class~.,data=train,cost=svmC,gamma=svmGam)
             pre <- predict(mod,test,probability=TRUE)
           },
           rpart={
             # Arboles de decision
             print("Se ejecuta la función de rpart")
             mod <- rpart(class~.,train)
             pre_prob <- predict(mod,test,type="prob")
             aux_tree <- data.frame(pre_prob)
             final <- test$class
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
    
    # Obtener el ACC
    aux <- sum(pre == test$class)/length(pre)
    #Acumular el valor ACC obtenido
    ACC_media <- ACC_media + aux
  }
  
  #Devolvemos el valor medio de AUC
  return(ACC_media/length(folds))
}

# ########################################
# CARGA Y PREPARACION DE DATOS DE DATOS
# ########################################

DatosCodificados <- read.csv(file="data_encoded.csv",header = TRUE)
DatosCodificados[paste("class")] <- test$class
DatosCodificados$class <- ifelse(DatosCodificados$class=="false-recurrence-events","NO","SI")
DatosCodificados$class <- as.factor(DatosCodificados$class)
DatosCodificados <- subset.data.frame(DatosCodificados,select = -X)

ACC_rd_3 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 3)
ACC_rd_5 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 5)
ACC_rd_7 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 7)
ACC_rd_10 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 10)
ACC_rd_13 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 13)
ACC_rd_15 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 15)
ACC_rd_17 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 17)
ACC_rd_20 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 20)
ACC_rd_30 <- crossVal(DatosCodificados,tamK = 10,metodo = "nnet",nnetTam = 30)
#Se aglutinan los datos utilizados y resultados para representarlos en una gráfica
size <- c(3,5,7,10,13,15,17,20,30)
ACC <- c(ACC_rd_3,ACC_rd_5,ACC_rd_7,ACC_rd_10,ACC_rd_13,ACC_rd_15,ACC_rd_17,ACC_rd_20,ACC_rd_30)
tabla_rn <- data.frame(ACC,size)
plot(tabla_rn$size,tabla_rn$ACC,main = "Comportamiento de nnet",xlab =  "Size",ylab ="ACC medio")
tabla_rn


#Solución para: Support Vector Machines
ACC_vm_1 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=1,svmC=1)
ACC_vm_2 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=1,svmC=50)
ACC_vm_3 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=1,svmC=60)
ACC_vm_4 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=1,svmC=80)
ACC_vm_5 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=1,svmC=90)
ACC_vm_6 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=50,svmC=1)
ACC_vm_7 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=60,svmC=50)
ACC_vm_8 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=80,svmC=60)
ACC_vm_9 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=90,svmC=80)
ACC_vm_10 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=100,svmC=90)
ACC_vm_11 <- crossVal(DatosCodificados,tamK = 10,metodo="e1071",svmGam=1000,svmC=100)

#Se aglutinan los datos utilizados y resultados para representarlos en una gráfica
ACC= c(ACC_vm_1,ACC_vm_2,ACC_vm_3,ACC_vm_4,ACC_vm_5,ACC_vm_6,ACC_vm_7,ACC_vm_8,ACC_vm_9,ACC_vm_10,ACC_vm_11)
gamma= c(1,1,1,1,1,50,60,80,90,100,1000)
cost= c(1,50,60,80,90,1,50,60,80,90,100)
tabla_vm <- data.frame(ACC,gamma,cost)
#install.packages("scatterplot3d")
require("scatterplot3d")
scatterplot3d(tabla_vm,main = "Comportamiento función svm",type = "h",box = FALSE,angle = 60)
tabla_vm


