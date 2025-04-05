# Barcamp 2024: Desplegando deep learning a producción con ONNX

Esta es la documentación del código de la charla de Desplegando Deep Learning a Producción con ONNX del evento Barcamp 2024 celebrado en Santiago, República Dominicana en Noviembre 2024.

Aquí se verán las instrucciones para instalar las dependencias y correr el código que se mostró en la charla.

## Instalación y configuración del environment.

**_Instalacion de Anaconda_**
En este apartado instalaremos la distribución de [Anaconda](https://www.anaconda.com/download) que es una variante del manejador de paquetes `conda` pero con repositorios configurados para liberias de ciencias de datos, machine learning e inteligencia artificial. Descargamos la version que vaya acorde a nuestro sistema operativo (Windows, macOS o Linux). Al descargarla seguimos las instrucciones según el sistema operativo que tengamos.

**_Instalación de docker_**

Para este codigo necesitaremos instalar docker ya que en el momento en que ejecutemos el servidor de producción de modelos de Machine Learning, Tensorflow Serving, utilizaremos la version en contenedor de docker que es mucho mas fácil de ejecutar y configurar.

Para descargar docker nos vamos a su [website oficial](https://www.docker.com) y descargamos la version compatible con el sistema operativo de nuestra preferencia y seguimos las instrucciones para instalarlo.

**_Configuración del environment e instalación de librerías y dependencias_**

Clonamos el repositorio con el código desde GitHub con el siguiente comando:

`git clone https://github.com/jpolanco91/barcamp2024-onnx.git`

En la terminal ejecutamos el siguiente comando de anaconda para configurar el environment del proyecto:

 `conda env create -f environment.yml`

Esto instalará todas las librerías requeridas por el proyecto para este funcionar de forma adecuada incluyendo el runtime de ONNX.

Luego de que se crea el environment con todas las librerías entonces debemos activarlo para poder correr el código de este proyecto, esto se hace con el siguiente comando:

`$ conda activate barcamp2024-onnx`

Luego de que ya no querramos seguir corriendo el código de este proyecto podemos desactivar el environment con el siguiente comando:

`$ conda deactivate`

Este comando desactiva los environment activos y no tenemos que especificar el nombre del environment cuando lo corremos.

## Corriendo el código del proyecto.

**_Corriendo el notebook de Jupyter_**

Jupyter es un sistema para correr código Python de forma interactiva. Se utiliza para investigación y desarrollo de modelos de ciencias de datos e inteligencia artificial, especialmente en el area de Machine Learning. Jupyter esta incluido en la lista de dependencias en `environment.yml` que es donde se configura el environment del codigo de este proyecto.

Luego de clonar el código del proyecto y activar el environment ejecutamos el comando 

`$ jupyter notebook predict-weight-variation-model.ipynb`

Ejecutamos el codigo celda por celda en Jupyter para poder ir creando los modelos y que tanto los modelos de Tensorflow y PyTorch se vayan creando y salvando. Cada celda de codigo en Jupyter esta acompañada de una descripción de lo que hace, así que solo es seguir las instrucciones.

**_Desplegando a producción el modelo creado por Tensorflow usando Tensorflow Serving_**

Luego de haber ejecutado el notebook de Jupyter en su totalidad se habra exportado el modelo de TensorFlow a un archivo que luego podemos desplegar a un servidor de producción como Tensorflow Serving. TensorFlow Serving es un servidor que permite desplegar modelos de machine learning a producción y que podamos pedirle inferencias (predicciones) al modelo a través de una API sea tipo REST o GRPC. En este caso utilizaremos una REST API ya que es mucho mas común en el mundo del desarrollo web.

Para desplegar a producción el modelo de TensorFlow usando TensorFlow Serving, primero debemos iniciar docker (seguir las instrucciones de acuerdo a su sistema operativo) y luego ejecutamos este comando:

`$ docker run --name=tensorflow-serving -t --rm -p 8501:8501 --mount 'type=bind,src=[/ruta/del/modelo],dst=/models/barcamp2024-onnx' -e MODEL_BASE_PATH=/models/barcamp2024-onnx -e MODEL_NAME=barcamp2024-model-tensorflow tensorflow/serving &`

Donde `[/ruta/del/modelo]` lo reemplazamos por la ruta donde se encuentra el modelo creado por TensorFlow (barcamp2024-model-tensorflow en este caso) en nuestro sistema operativo.

Si no ocurre ningún error al correr el comando anterior, el servidor de TensorFlow serving desplegara nuestro modelo en el puerto 8501 y el mismo se encuentra en el siguiente endpoint RESTful: http://localhost:8501/v1/models/barcamp2024-model-tensorflow:predict

Donde `predict` sería el action que realiza la inferencia al modelo (predicción).

Podemos hacer requests a esa url mediante `curl` o herramientas como Postman y el payload de dicho request seria de esta forma:

``` 
{
    "instances": [[0.12195122, 0.39797395, 0.51170312, -1.0272263, -0.01766304, 6.72727273, 8.625, -0.05361132, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
}
```

Donde `instances` serian los features que corresponderian a la predicción que se quiere hacer con el modelo y los mismos se presentan en un array anidado y cada feature debe estar pre-procesado con el mismo procedimiento que se hizo en el notebook de Jupyter para que la predicción sea confiable y cada uno debe ir en el mismo orden en que se muestra en el data frame de pandas que también esta en el notebook.

Al hacer el request con el payload anterior la API devolverá un objeto JSON con las predicciones en una propiedad llamada `predictions`

**_Desplegando a producción el modelo creado por PyTorch usando TorchServe_**

Luego de haber ejecutado el notebook de Jupyter en su totalidad se habra exportado el modelo de PyTorch a un archivo que luego podemos desplegar al servidor de producción de PyTorch llamado TorchServe. TorchServe es un servidor que permite desplegar modelos de machine learning a producción al igual que TensorFlow serving y que podamos pedirle inferencias (predicciones) al modelo a través de una API sea tipo REST o GRPC. En este caso utilizaremos una REST API ya que es mucho mas común en el mundo del desarrollo web.

En este caso para TorchServe no necesitaremos docker para su ejecución aunque existe una version en contenedor de docker si así lo desea. En este ejemplo usaremos el TorchServe que se instalo junto con las dependencias del proyecto en `environment.yml` a través de `conda`.

Estando dentro de la carpeta de codigo del proyecto, primero tenemos que convertir el archivo serializado `.pth` que generamos en Jupyter con PyTorch, el cual contiene el modelo de deep learning con todos los pesos (weights) pre-entrenados con la data de training. Esto porque para TorchServe desplegar el modelo a producción necesita estar en el formato Model Archiver (MAR) de TorchServe.

Para convertir el modelo serializado en el formato de model archiver (MAR) usamos la herramienta `torch-model-archiver` incluida en las dependencias del proyecto en `environment.yml`. En nuestra Terminal o linea de comandos, dentro de la carpeta donde se encuentra el codigo del proyecto, ejecutamos el siguiente comando:

`$ torch-model-archiver --model-name barcamp2024-model-pytorch --version 1.0 --model-file barcamp2024-pytorch-model.py --serialized-file barcamp2024-model-pytorch.pth --extra-files preprocessing_pipelines.py  --handler barcamp2024-pytorch-model-handler.py`

Si no se presenta ningún error, esto generará un archivo con el nombre `barcamp2024-model-pytorch.mar` el cual es el modelo ya empaquetado para poder desplegarlo a TorchServe.

Para desplegar el modelo al servidor de TorchServe (producción) ejecutamos el siguiente comando:

`$ torchserve --start --ncs --ts-config config.properties --model-store . --models barcamp2024-model-pytorch.mar`

Si no ocurre ningún error esto ejecutara el servidor de TorchServe con el modelo ya desplegado. Este estará desplegado en un endpoint tipo REST en la siguiente URL: http://localhost:8080/predictions/barcamp2024-model-pytorch

En este caso TorchServe usa el puerto 8080 para servir los modelos, mientras que el puerto 8081 lo utiliza para tareas administrativas. Para inferencia o predicciones de modelos siempre usamos el 8080. El payload con los features que le enviaremos para una predicción o inferencia será de esta forma:

```
{
    "age": 22,
    "gender": "F",
    "current_weight": 155,
    "bmr": 1300.0,
    "daily_calories_consumed": 1600.0,
    "daily_caloric_surplus_deficit": 90.0,
    "duration": 75.0,
    "stress_level": 90,
    "physical_activity_level": "Very Active",
    "sleep_quality": "Excellent"
}
```

Notese que a diferencia de TensorFlow Serving en TorchServe podemos enviar un payload mas descriptivo y fácil de entender ya que TorchServe permite personalizar aun mas las tareas de inferencia y predicciones permitiendo aceptar el payload justo como lo hicimos en el Jupyter notebook y ya en uno de los archivos que empaquetamos en el formato MAR ya le enviamos el codigo fuente de como pre-procesar este payload para que al final el modelo lo acepte en el formato adecuado para hacer la inferencia/prediccion. Esto se define en el archivo de handler `barcamp2024-pytorch-model-handler.py`

Si no hay errores luego de hacer el request con el payload de arriba, debería de mostrarnos la predicción basado en los features enviados por el request.

**_Desplegando a producción los modelos creados tanto por TensorFlow como por  PyTorch usando ONNX_**

Aquí entramos al punto central del código de esta charla. Anteriormente vimos como para cada librería tuvimos que configurar un runtime/servidor diferente según la librería donde hayamos creado el modelo.

Con ONNX solo necesitaremos 1 único runtime/servidor para poder hacer nuestra inferencia al modelo independientemente de donde hayamos entrenado el modelo.

ONNX se instaló junto con las demás dependencias del proyecto que se especificaron en `environment.yml`

Para ejecutar los modelos de ONNX en el runtime del mismo nombre, he creado un programa de linea de comando llamado `onnx-entry-point.py` que básicamente podremos mandarle el dato que queremos inferir o predecir por linea de comando y este nos devolverá el resultado de dicha predicción por linea de comando. El programa recibe 2 parámetros:

 - El modelo que queremos: Si es PyTorch el parametro es `-p true` o `--pytorch true`; si es TensorFlow el parametro es `-t true` o `--tensorflow true`.

 - El datapoint para el cual queremos sacar la inferencia: el parámetro es aquí `-d '{objeto json con el datapoint que queremos inferir}'`o `--datapoint '{objeto json con el datapoint que queremos inferir}'`

ONNX puede integrarse con cualquier framework web o API de nuestra preferencia. Por falta de tiempo no pude crear una web app o API de ejemplo para integrar ONNX, pero si desean hacerlo pueden tomar el archivo `onnx-entry-point.py` como base para ello.

Luego de haber ejecutado el notebook de Jupyter por completo, los modelos tanto de PyTorch como de TensorFlow se habrán exportado al formato `.onnx` con sus respectivos nombres: `barcamp2024-model-pytorch.onnx` y `barcamp2024-model-tensorflow.onnx` respectivamente. Con TensorFlow se utilizo la librería `tf2onnx` (incluida en `environment.yml`) mientras que PyTorch ya tiene integrado en su codigo fuente un mecanismo para exportar el modelo al formato ONNX.

Para desplegar el modelo en ONNX usando el modelo generado por PyTorch usamos este comando: `python onnx-entry-point.py -d '{"age": 22, "gender": "F", "current_weight": 155, "bmr": 1300.0, "daily_calories_consumed": 1600.0, "daily_caloric_surplus_deficit": 90.0, "duration": 75.0, "stress_level": 90, "physical_activity_level": "Very Active", "sleep_quality": "Excellent"}' -p true`

Para desplegar el modelo en ONNX usando el modelo generado por TensorFlow usamos este comando: `python onnx-entry-point.py -d '{"age": 22, "gender": "F", "current_weight": 155, "bmr": 1300.0, "daily_calories_consumed": 1600.0, "daily_caloric_surplus_deficit": 90.0, "duration": 75.0, "stress_level": 90, "physical_activity_level": "Very Active", "sleep_quality": "Excellent"}' -t true`

Nótese en los comandos anteriores que gracias a tener un único runtime o servidor (ONNX) tenemos prácticamente el mismo tipo comando, la diferencia es el tipo de modelo a seleccionar (PyTorch y TensorFlow), esto incluso facilita la creación de APIs web (REST, GRPC u otro tipo) ya que con un único runtime podemos crear un único formato de endpoints y la diferencia seria la configuración del tipo de modelo a seleccionar, por lo que no tenemos que estar atados a una única tecnología para desarrollar y desplegar modelos de machine learning en producción.
