# sih-faceRecognition
Done as a part of Smart India Hackathon 2020
## Face Recognition
The face is one of the easiest ways to distinguish the individual identity
of each other. Face recognition is a **personal identification system** that uses
personal characteristics of a person to identify the person's identity. Human face
recognition procedure basically consists of two phases, namely **face detection**,
where this process takes place very rapidly in humans, except under conditions
where the object is located at a short distance away, the next is the **introduction**,
which recognize a face as individuals. We have designed the face recognition
system by using **HOG (Histogram of Oriented Gradient)**. HOG is one of the
finest object detection method which is the most lightest and accurate for
detecting. Once a face is detected we train a Machine Learning algorithm to find
the face landmarks (68 Specific points). Every human face have **128 unique
measurements (embedding)** in it. To find the embeddings of a face, we use a
trained deep convolutional neural network which uses a **Triplet Loss
function** to encode, where 3 images are taken – two of the same person and one
of the different. The detected face is then compared with already existing target
faces where the **best match** is found and recognised. Coded in Python!
## Requirements
* OpenCV  (Open source Computer Vision) library for image processing
* NumPy Package for handling array data 
* Dlib library for implementing Machine learning algorithm which is used here and also for Image Classification.
* Glob module to specify the filename.
## Face Detection Module
* Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images.
* We’re going to use a method invented in 2005 called Histogram of Oriented Gradients — or just HOG for short. 
* The purpose of using HOG is, it detects the faces more accurately than any other detection algorithms.
* HOG runs lightly on  CPU
## Working of HOG
* To find faces in an image, we’ll start by making our image black and white.
* Then we’ll look at every single pixel in our image one at a time. For every single pixel, we want to look at the pixels that directly surrounding it:<br />
![HOG](https://miro.medium.com/max/875/1*RZS05e_5XXQdofdRx1GvPA.gif)<br />
* Our goal is to figure out how dark the current pixel is compared to the pixels directly surrounding it. Then we want to draw an arrow showing in which direction the image is getting darker:<br/>![HOG](https://miro.medium.com/max/625/1*WF54tQnH1Hgpoqk-Vtf9Lg.gif)<br/>
* If you repeat that process for every single pixel in the image, you end up with every pixel being replaced by an arrow. These arrows are called gradients and they show the flow from light to dark across the entire image:<br/>
![HOG](https://miro.medium.com/max/875/1*oTdaElx_M-_z9c_iAwwqcw.gif)<br />
* The end result is we turn the original image into a very simple representation that captures the basic structure of a face in a simple way:<br />
![HOG](https://miro.medium.com/max/875/1*uHisafuUw0FOsoZA992Jdg.gif)<br/>
* To find faces in this HOG image, all we have to do is find the part of our image that looks the most similar to a known HOG pattern that was extracted from a bunch of other training faces:<br />![HOG](https://miro.medium.com/max/875/1*6xgev0r-qn4oR88FrW6fiA.png)<br/>

## Face Recognition with Embeddings
We need to be able to recognize faces in milliseconds, not hours.
What we need is a way to extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements.
 For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc. 
 Deep learning does a better job than humans at figuring out which parts of a face are important to measure.The solution is to train a Deep Convolutional Neural Network
We are going to train it to generate 128 measurements for each face.<br />
The training process works by looking at 3 face images at a time:
* 1-Load a training face image of a known person.
* 2-Load another picture of the same known person.
* 3-Load a picture of a totally different person
Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2 are slightly closer while making sure the measurements for #2 and #3 are slightly further apart.<br/>
![HOG](https://miro.medium.com/max/518/1*AbEg31EgkbXSQehuNJBlWg.png)
![HOG](https://miro.medium.com/max/875/1*xBJ4H2lbCMfzIfMrOm9BEQ.jpeg)
![HOG](https://miro.medium.com/max/875/1*n1R8VMyDRw3RNO3JULYBpQ.png)<br/>
* Machine learning people call the 128 measurements of each face an embedding. The idea of reducing complicated raw data like a picture into a list of computer-generated numbers comes up a lot in machine learning
 * OpenFace already trained the model and they published several trained networks which we can directly use.
So all we need to do ourselves is run our face images through their pre-trained network to get the 128 measurements for each face. 
All that we care is that the network generates nearly the same numbers when looking at two different pictures of the same person.<br/>
![HOG](https://miro.medium.com/max/875/1*6kMMqLt4UBCrN7HtqNHMKw.png)<br/>
The last step is actually the easiest step in the whole process. All we have to do is find the person in our database of known people who has the closest measurements to the detected face.
This can be done by using any basic machine learning classification algorithm. 
Here we use a simple linear SVM classifier.
All we need to do is train a classifier that can take in the measurements from a new detected face and tells which known person is the closest match. Running this classifier takes milliseconds. The result of the classifier is the name of the person!
## References
[https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



