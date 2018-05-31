#Cargar librerias
library("keras", lib.loc="~/R/win-library/3.4")
library("tensorflow", lib.loc="~/R/win-library/3.4")



# Definición de los parámetros comunes --------------------------------------------------------------
# la "L" se pone para que sea de tipo Integer
batch_size <- 256L
# this is the size of our encoded representations
original_dim <- 784L
latent_dim <- 784L
encoding_dim <- 32L
#epochs <- 50L

# Definicion del modelo --------------------------------------------------------

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


#Let's also create a separate encoder model:
#Tambien se puede presentar por separado la salida de la codficacion 

# this model maps an input to its encoded representation
# Da los datos codificados (disminuye el tamaño de la entrada)
encoder <- keras_model(input_img, encoded)


#As well as the decoder model:
# create a placeholder for an encoded (32-dimensional) input
# Se crea la entrada directamente proviniente de una codificacion anterior
encoded_input <- layer_input(shape = c(encoding_dim))

# retrieve the last layer of the autoencoder model
# Recupera la ultima capa del modelo de autoencoder definido anteriormene
decoder_layer = get_layer(autoencoder,index=-1)


# create the decoder model
# Definir la forma de decofificacion
decoder <- keras_model(encoded_input, decoder_layer(encoded_input))


# Para compilar el modelo se necesita un optimizador y la perdida
autoencoder  %>% compile(optimizer = 'adadelta', loss = 'binary_crossentropy')


# Conjutos de datos y aplicacion del autoencoder---------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255

# ??? que es esto, entiendo que está cargando los datos al modelo que va a aplicar 
x_train <- x_train %>% apply(1, as.numeric) %>% t()
x_test <- x_test %>% apply(1, as.numeric) %>% t()


# Trains the model for a given number of epochs (iterations on a dataset).
autoencoder %>% fit (x_train, x_train,
                     epochs=50,
                     batch_size=256,
                     shuffle=TRUE,
                     validation_data=list(x_test, x_test))