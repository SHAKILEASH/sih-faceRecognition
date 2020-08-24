# sih-faceRecognition
Done as a part of Smart India Hackathon 2020
# Face Recognition
The face is one of the easiest ways to distinguish the individual identity
of each other. Face recognition is a personal identification system that uses
personal characteristics of a person to identify the person's identity. Human face
recognition procedure basically consists of two phases, namely face detection,
where this process takes place very rapidly in humans, except under conditions
where the object is located at a short distance away, the next is the introduction,
which recognize a face as individuals. We have designed the face recognition
system by using HOG (Histogram of Oriented Gradient). HOG is one of the
finest object detection method which is the most lightest and accurate for
detecting. Once a face is detected we train a Machine Learning algorithm to find
the face landmarks (68 Specific points). Every human face have 128 unique
measurements (embedding) in it. To find the embeddings of a face, we use a
trained deep convolutional neural network which uses a Triplet Loss
function to encode, where 3 images are taken – two of the same person and one
of the different. The detected face is then compared with already existing target
faces where the best match is found and recognised.
