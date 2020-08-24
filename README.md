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
faces where the **best match** is found and recognised.
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
* Our goal is to figure out how dark the current pixel is compared to the pixels directly surrounding it. Then we want to draw an arrow showing in which direction the image is getting darker:![HOG](https://miro.medium.com/max/625/1*WF54tQnH1Hgpoqk-Vtf9Lg.gif)<br/>
* If you repeat that process for every single pixel in the image, you end up with every pixel being replaced by an arrow. These arrows are called gradients and they show the flow from light to dark across the entire image:<br/>
![HOG](https://miro.medium.com/max/875/1*oTdaElx_M-_z9c_iAwwqcw.gif)<br />
* The end result is we turn the original image into a very simple representation that captures the basic structure of a face in a simple way:<br />
![HOG](https://miro.medium.com/max/875/1*uHisafuUw0FOsoZA992Jdg.gif)

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
Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2 are slightly closer while making sure the measurements for #2 and #3 are slightly further apart.
* Machine learning people call the 128 measurements of each face an embedding. The idea of reducing complicated raw data like a picture into a list of computer-generated numbers comes up a lot in machine learning
 * OpenFace already trained the model and they published several trained networks which we can directly use.
So all we need to do ourselves is run our face images through their pre-trained network to get the 128 measurements for each face. 
All that we care is that the network generates nearly the same numbers when looking at two different pictures of the same person.



