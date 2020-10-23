# Malaria_Net
## Background
Severe malaria is commonly misdiagnosed in Africa, leading to a failure to treat other life-threatening illnesses. In malaria-endemic areas, parasitemia does not ensure a diagnosis of severe malaria because parasitemia can be incidental to other concurrent disease. The detection of malarial retinopathy is a candidate diagnostic test for cerebral malaria. Malarial retinopathy consists of a set of retinal abnormalities that is unique to severe malaria and common in children with cerebral malaria

The detection and assessment of leakage in retinal fluorescein angiogram images is important for the management of a wide range of retinal diseases. We have developed a framework that can automatically detect three types of leakage (large focal, punctate focal and vessel segment leakage).

![alt text](https://github.com/samadon1/Malaria_Net/blob/main/Capture11.JPG " Working app screen capture ")


## The model
This is a deep learning model created for the purpose of computer aided diagnosis of papilledema, vessel leakage, punctuate leakage, focal leakage and normal retinal images, a set of retinal abnormalities that is unique to severe malaria which is common in children with cerebral malaria.

Becuase of the smaller size of the dataset ( 53 training images and 15 validation images) stored in [google cloud storage](https://cloud.google.com/storage), by using transfer learning, the model was pretrained on the xception pretrained model.
The model was trained with google cloud TPUs [google cloud TPUs](https://cloud.google.com/tpu)

## Explainability
To know why the neural network made a decision, the results of the model can be understood  and visualized by tf_explain library.

## Deployment
The streamlit-ready app will be containerized with [docker and deployed on kubernestes](https://www.docker.com/products/kubernetes)

