####################
#Carga de la librerias necesarias
####################

library("rminer")
library("keras")
library("tensorflow")
library("reticulate")

# ####################
# CARGA Y PREPARACION DE DATOS DE DATOS
# ####################

#' LEER CSV
#' Es un conjunto de datos etiquetados. Objeto con 500 registros de 8 variables
#' LEER CSV

Datos <- read.table(file="datos_campus_virtual.txt",header=TRUE)

# convertido a datos categoricos
# for(unique_value in unique(Datos$feno)){
#   Datos[paste("feno", unique_value, sep = ".")] <- ifelse(Datos$feno == unique_value, 1, 0)
# }

# for(unique_value in unique(Datos$grado)){
#   Datos[paste("grado", unique_value, sep = ".")] <- ifelse(Datos$grado == unique_value, 1, 0)
# }


#' PRUNE THE DATA
#' En el caso de que existas datos no relevantes o registros con valores nulos
#' PRUNE THE DATA


# Información del conjunto de datos
summary(Datos)

# Metedo de validacion interna holdout 
# Creacion de subconjutnos de entrenamiento (334) y test (166)
Division <- holdout(y=Datos$recid)

# Obtener conjuntos X,Y de test y entrenamiento
train <- Datos[Division$tr,]
x_train <- subset.data.frame(train,select = -recid)
y_train <- subset.data.frame(train,select = recid)

test <- Datos[Division$ts,]
x_test <- subset.data.frame(test,select = -recid)
y_test <- subset.data.frame(test,select = recid)


# Comvertirlo en tipo matriz
x_train_matrix = data.matrix(x_train)
y_train_matrix = data.matrix(y_train)

x_test_matrix = data.matrix(x_test)
y_test_matrix = data.matrix(y_test)

print(cat('Conjunto de entrenamiento. Número de columna: ',ncol(x_train_matrix),' Número de filas: ',nrow(x_train_matrix)))
print(cat('Conjunto de test Número de columna: ',ncol(x_test_matrix),' Número de filas: ',nrow(x_test_matrix)))


# ####################
#    AUTOENCODER
# ####################

# Variables de inicio

# Porque el número de columnas del conjunto de variables es 7
original_dim <- 7L 
# Reducir el número en la capa de codificación
encoding_dim <- 4L 
# Número de neurona de salida
latent_dim <- 7L 

# **********************
# Definicion del modelo 
# **********************

# this is our input placeholder
# Se establece la entrada como un vector con un numero de neuronas de original_dim
input <- layer_input(shape = c(original_dim))

# "encoded" is the encoded representation of the input
# La capa de codificación, donde se le pasa la capa de entrada, dimesion (numero de neuronas), funcion de activacion
encoded<- layer_dense(input,encoding_dim , activation = "relu")

#Generamos el modelo de condificacion
model_enconded <- keras_model(input, encoded)

# La capa de decodificacion, vuelve a tener las dimensiones de la entrada existe una perdida de informacion con respecto a la entrada
# Como prámetros se utiliza la salida de la capa de codificacion,tamaño de la capa de decodificacion (latent_dim), funcion de activacion (sigmoid) 
decoded<- layer_dense(encoded, latent_dim , activation = "sigmoid")

#Generamos el modelo de autoencoder
# Se presenta el autoenconder entero, dentro de el codifica y decodifica
# Donde le proporciona la entrada y el decode definido que tira del encode tambien definido
model_autoencoder <- keras_model(input, decoded)

# **********************
# Compilacion del modelo 
# **********************


# Para compilar el modelo se necesita un optimizador y la perdida
model_autoencoder  %>% compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

# Información sobre el autoencoder
summary(model_autoencoder)


# **********************
# Entrenamiento del modelo 
# **********************

#history <- autoencoder %>% fit(x_train_matrix, x_train_matrix,epochs=50,batch_size=256,shuffle=TRUE)
history <- model_autoencoder %>% fit(x_train_matrix, x_train_matrix,epochs=50,batch_size=256)

plot(history)

pesos_autoencoder <- get_weights(model_autoencoder)
pesos_encoded <- get_weights(model_enconded)


