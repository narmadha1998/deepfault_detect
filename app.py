from flask import Flask
from flask import request, redirect, url_for, render_template
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required

app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pgadmin@localhost/defect_detection'
app.config['SECRET_KEY'] = 'super-secret'
app.config['SECURITY_REGISTERABLE'] = True

# app.debug = True
#db = SQLAlchemy(app)


# Define models
'''roles_users = db.Table('roles_users',
        db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
        db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))'''

# Setup Flask-Security
'''user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)'''
# Create a user to test with
'''@app.before_first_request
def create_user():
    db.create_all()
    user_datastore.create_user(email='narmadhanarmu98@gmail.com', password='12345')
    db.session.commit()'''
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile/')
@login_required
def profile(email):
    user = User.query.filter_by(email=email).first()
    return render_template('profile.html', user=user)
image_path = 'static/input/'
@app.route('/predict',methods=['POST'])
def predict():
    import os
    model_input = request.form.get("input_image")
    print(model_input)
    image_name = os.path.join(image_path,model_input)
    print(image_name)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Convolution2D as Con2D,MaxPooling2D
    from tensorflow.keras.layers import Activation,Dropout,Flatten,Dense
    from tensorflow.keras import backend as k
    import numpy as np
    from tensorflow.keras.preprocessing import image
    import os
    img_width,img_height= 150,150
    # data_dir = "data/"
    # train_data_dir=data_dir+"train/"
    # validation_data_dir=data_dir+"test/"
    # nb_train_samples= 1000
    # nb_validation_samples= 100
    # epochs= 4
    # batch_size= 100
    if k.image_data_format() == 'channels_first':
        input_shape = (3,img_width,img_height)
    else:
        input_shape = (img_width,img_height,3)
    #     train_datagen = ImageDataGenerator(
    #         rescale=1. / 255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    # train_generator = train_datagen.flow_from_directory(
    #         train_data_dir,
    #         target_size=(img_width, img_height),
    #         batch_size = batch_size,
    #         class_mode='binary')
    # validation_generator = train_datagen.flow_from_directory(
    #         validation_data_dir,
    #         target_size=(img_width, img_height),
    #         batch_size = batch_size,
    #         class_mode='binary')
    model = Sequential()
    model.add(Con2D(32,(3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Con2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Con2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=nb_train_samples // batch_size,
    #         epochs=epochs,
    #         validation_data= validation_generator,
    #         validation_steps=nb_validation_samples // batch_size)
    
    # model.save("final-model.h5")
    # model.save_weights('final_pro.h5')

    model.load_weights("final_pro.h5")
    import cv2
    import matplotlib
    import numpy as np
    import scipy
    import scipy.io as sio
    import imutils
    import os
    #import mahotas as mt
    from imutils import contours
    from sklearn.cluster import KMeans
    from sklearn.cluster import spectral_clustering
    from sklearn.neural_network import MLPClassifier
    from tensorflow.keras.models import load_model
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    p=1;
    print ("*****Testing*****")
    # model.load_weights('final_proweights.h5')
    # print(model.summary())

    # Loading image
    #os.chdir('E:/code/2018/opencv-python/lung_code_python/TestImages');   ## change here the test image path
    img = cv2.imread(image_name)
    import os
    # print(os.path.exists('project/data/test/defective/one.jpg'))
    #os.chdir('E:/code/2018/opencv-python/lung_code_python');## change here to original code path
    # cv2.imshow('Input Image',img)
    # cv2.waitKey(2)

    ##Gaussian Blurring
    kernel = np.ones((7,7),np.float32)/25
    img1 = cv2.filter2D(img,-1,kernel)
    #cv2.imshow('Gaussian Image',img1)

    ## Bilateral Filter for Edge Enhancement
    img3 = cv2.bilateralFilter(img1,9,75,75)
    #cv2.imshow('Bilateral Filtered Image',img3)


    ## RGB to Gray conversion
    GRAY_Img = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('GRAY Image',GRAY_Img)

    Data2Ext=GRAY_Img;
    #cv2.imwrite('ImageRedist.jpg',Data2Ext);


    roi1=GRAY_Img;
    r,c=roi1.shape;
    if p==1:
        roi = roi1.reshape((roi1.shape[0] * roi1.shape[1], 1))

    ## KMEANS clustering
    imgkmeans = KMeans(n_clusters=2, random_state=0);
    imgkmeans.fit(roi);
    label_values=imgkmeans.labels_;
    Label_reshped = np.reshape(label_values,(roi1.shape[0] ,roi1.shape[1]));

    segmentregions=roi1;
    blobregions=roi1;

    rows,cols = roi1.shape;
    # Thresholding for segmentation
    for i in range(0,rows):
        for j in range(0,cols):
            pixl=Label_reshped[i,j];
            if pixl==0:
                segmentregions[i,j]=0;
                
            else:
                segmentregions[i,j]=255;


    #plt.imshow(segmentregions)

    #segmentregions = cv2.resize(segmentregions , (1,(150,150)))
    size =150
    segmentregions1 = np.array(segmentregions)
    segmentregions1.resize(1,150,150,3)


    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    rslt= model.predict(segmentregions1)
    print(rslt)
    if rslt[0][0] == 0 :
        prediction="defective"
    else:
        prediction="non-defective"
    print(prediction)
    return render_template("result.html",rslt=rslt,prediction=prediction,name=model_input)



@app.route('/post_user', methods=['POST'])
def post_user():
    user = User(request.form['username'], request.form['email'])
    #db.session.add(user)
    #db.session.commit()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run()
