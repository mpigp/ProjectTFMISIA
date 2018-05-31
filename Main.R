####################
#Carga de la librerias necesarias
####################

library("rminer")
library("keras")
library("tensorflow")
library("reticulate")

#K <- keras::backend()



# ####################
# CARGA Y PREPARACION DE DATOS DE DATOS
# ####################

#' LEER CSV
#' Es un conjunto de datos etiquetados. Objeto con 500 registros de 8 variables
#' LEER CSV

Datos <- read.table(file="datos_campus_virtual.txt",header=TRUE)
 
#' PRUNE THE DATA
#' En el caso de que existas datos no relevantes o registros con valores nulos
#' PRUNE THE DATA


# Información del conjunto de datos
#summary(Datos)

# Metedo de validacion interna holdout 
# Creacion de subconjutnos de entrenamiento (334) y test (166)
Division <- holdout(y=Datos$recid)

# Ya el metodo de partición da los indices del objeto se deben obtener dichos registros de los indices
x_train <- Datos[Division$tr,]
x_test <- Datos[Division$ts,]

# Comvertirlo en tipo matriz
x_train_matrix = data.matrix(x_train)
x_test_matrix = data.matrix(x_test)

print(cat('Conjunto de entrenamiento. Número de columna: ',ncol(x_train_matrix),' Número de filas: ',nrow(x_train_matrix)))
print(cat('Conjunto de test Número de columna: ',ncol(x_test_matrix),' Número de filas: ',nrow(x_test_matrix)))


# Esto sería como en el ejemplo, pero no entiendo para que lo hace 
#x_train_2 <- x_train_matrix %>% apply(1, as.numeric) %>% t()
#x_test_2 <- x_test_matrix %>% apply(1, as.numeric) %>% t()

#print(cat('Conjunto de entrenamiento 2. Número de columna: ',ncol(x_train_2),' Número de filas: ',nrow(x_train_2)))
#print(cat('Conjunto de test 2. Número de columna: ',ncol(x_test_2),' Número de filas: ',nrow(x_test_2)))


# ####################
#    AUTOENCODER
# ####################

# Variables de inicio

# Porque el número de columnas del conjunto de variables es 8
original_dim <- 8L #334L
# Reducir el número en la capa de codificación
encoding_dim <- 4L #32L
# Número de neurona de salida
latent_dim <- 8L #334L

# **********************
# Definicion del modelo 
# **********************

# this is our input placeholder
# Se establece la entrada como un vector con un numero de neuronar de original_dim = 784L
input_img <- layer_input(shape = c(original_dim))

# "encoded" is the encoded representation of the input
# La capa de codificación, donde se le pasa la capa de entrada, dimesion (numero de neuronas), funcion de activacion (Rectificador (redes neuronales))
encoded<- layer_dense(input_img,encoding_dim , activation = "relu")

# "decoded" is the lossy reconstruction of the input
# La capa de decodificacion, vuelve a tener las dimensiones de la entrada existe una perdida de informacion con respecto a la entreda
# Como prámetros se utiliza la salida de la capa de codificacion,tamaño de la capa de decodificacion (latent_dim), funcion de activcion (sigmoid) 
decoded<- layer_dense(encoded, latent_dim , activation = "sigmoid")

# this model maps an input to its reconstruction
# Se presenta el autoenconder entero, dentro de el codifica y decodifica
# Donde le proporciona la entrada y el decode definido que tira del encode tambien definido
#  Con keras_model ? se está haciendo una llamada a la API que proporiona Keras
autoencoder <- keras_model(input_img, decoded)

# **********************
# Compilacion del modelo 
# **********************


# Para compilar el modelo se necesita un optimizador y la perdida
autoencoder  %>% compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

# Información sobre el autoencoder
summary(autoencoder)


# **********************
# Entrenamiento del modelo 
# **********************

history <- autoencoder %>% fit (x_train_matrix, x_train_matrix,
                     epochs=50,
                     batch_size=256,
                     shuffle=TRUE,
                     validation_data=list(x_test_matrix, x_test_matrix))

plot(history)

####################
# Definición de un encoded
####################

# Definición de la capa de entrada
input_layer <- layer_input(shape = c(original_dim))

# Definición de la capa codificadora
encoded_layer<- layer_dense(input_layer,encoding_dim , activation = "relu")

# Se crea la parte codificadora
encoder_model <- keras_model(input_layer, encoded_layer)

# Compilar el modelo
encoder_model  %>% compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

# Aplicar modelo a los datos anteriormente definidos
encoder_model %>% fit (x_train_matrix,x_train_matrix,
                    epochs=50,
                     batch_size=256,
                     shuffle=TRUE,
                     validation_data= list(x_test_matrix,x_test_matrix))
