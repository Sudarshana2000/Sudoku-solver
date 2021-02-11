'''Create and save model for digit recognition'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense,Dropout

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

class SudokuNet:
    
    @staticmethod
    def build(width,height,depth,classes):
        # w,h,d,c = 28 pixels X 28 pixels X 1 grayscale channel (B&W) X 10 digits 0_9
        model=Sequential()
        inputshape=(height,width,depth)
        
        #first set of convolution->relu->pool layers
        model.add(Conv2D(32, (5,5), padding="same",input_shape=inputshape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #second set of convolution->relu->pool layers
        model.add(Conv2D(32, (3,3), padding="same",input_shape=inputshape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #first set of fully connected layers with 50% dropout -> relu
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        #second set of fully connected layers with 50% dropout -> relu
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        #softmax classifier: no. of outputs = no. of classes
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
    
ALPHA=0.001 #initial learning rate
EPOCH=10
BatchSize=128

print("Loading MNIST Datasets...")
((trainData,trainLabels),(testData,testLabels))=mnist.load_data()
trainData=trainData.reshape((trainData.shape[0],trainData.shape[1],trainData.shape[2],1))
testData=testData.reshape((testData.shape[0],testData.shape[1],testData.shape[2],1))
trainData=trainData.astype("float32")/255.0
testData=testData.astype("float32")/255.0
lb=LabelBinarizer()
trainLabels=lb.fit_transform(trainLabels)
testLabels=lb.transform(testLabels)

print("Compiling model...")
# compile model using Adam optimizer
optimizer=Adam(learning_rate=ALPHA)
# building model...
model=SudokuNet.build(width=28, height=28, depth=1, classes=10)
#if the focus is only on recognizing binary numbers 0 and 1, then use loss="binary_crossentropy"
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

print("Training network...")
model.fit(trainData,trainLabels,BatchSize,EPOCH,1,validation_data=(testData,testLabels))
print("Evaluating network...")
res=model.predict(testData)
print(classification_report(testLabels.argmax(axis=1), res.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

print("Saving model in HDF5 format...")
model.save("F:/models/digit_classifier.h5",save_format="h5")
