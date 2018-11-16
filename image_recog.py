
import os
import sys
import time

import dlib
import cv2
import numpy as np
import pickle
import tensorflow as tf
from scipy import misc
from packages import facenet, detect_face


def recognizer(images_path, 
			pretrain_model='./models/20180408-102900', 
			classifier='./class/classifier.pkl', 
			npy_dir='./packages', 
			train_img_dir='./datasets'):
	with tf.Graph().as_default():
		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
		gpu_options = tf.GPUOptions(allow_growth=True)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
			img_options = {'minsize':20,
						'threshold':[0.6, 0.7, 0.7],
						'factor':0.709,
						'margin':44,
						'frame_interval':3,
						'batch_size':1000,
						'image_size':182,
						'input_image_size':160 }

			HumanNames = os.listdir(train_img_dir)
			HumanNames.sort()

			print('Loading model...')
			facenet.load_model(pretrain_model)

			images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
			embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
			embedding_size = embeddings.get_shape()[1]

			classifier_exp = os.path.expanduser(classifier)
			with open(classifier_exp, 'rb') as infile:
				(model, class_names) = pickle.load(infile)
			
			c = 0
			print('Facial Recognition Starting...')
			#win = dlib.image_window()
			for img in images_path:
				prevTime = 0
				frame = cv2.imread(img, 0)
				frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

				#frame = dlib.load_rgb_image(img)
				#frame = dlib.resize_image(img, 0.5, 0.5)

				curTime = time.time() + 1
				timeF = img_options['frame_interval']

				if (c%timeF == 0):
					find_results = []

					if frame.ndim == 2:
						frame = facenet.to_rgb(frame)
					frame = frame[:, :, 0:3]
					bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, img_options['threshold'], img_options['factor'])
					nrof_faces = bounding_boxes.shape[0]
					print('Face Detected: %d' % nrof_faces)
					if nrof_faces > 0:
						det = bounding_boxes[:, 0:4]
						img_size = np.asarray(frame.shape)[0:2]

						cropped = []
						scaled = []
						scaled_reshape = []
						bb = np.zeros((nrof_faces, 4), dtype=np.int32)

						for i in range(nrof_faces):
							emb_array = np.zeroes((1, embedding_size))

							bb[i][0] = det[i][0]
							bb[i][1] = det[i][1]
							bb[i][2] = det[i][2]
							bb[i][3] = det[i][3]

							if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        		print('face is too close')
								continue

							cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
							cropped[i] = facenet.flip(cropped[i], False)
							scaled.append(misc.imresize(cropped[i], (img_options['image_size'], img_options['image_size']), interp='bilinear'))
							scaled[i] = cv2.resize(scaled[i], (img_options['input_image_size'], img_options['input_image_size']), interpolation=cv2.INTER_CUBIC)
							scaled[i] = facenet.prewhiten(scaled[i])
							scaled_reshape.append(scaled[i].reshape(-1, img_options['input_image_size'], img_options['input_image_size'], 3))
							feed_dict = {img_options['images_placeholder']: scaled_reshape[i], img_options['phase_train_placeholder']: False}
							emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
							predictions = model.predict_proba(emb_array)
							print(predictions)
							best_class_indices = np.argmax(predictions, axis=1)
							best_class_probabilites = predictions[np.arrange(len(best_class_indices)), best_class_indices]
							print(best_class_probabilities)
							cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 1)

							text_x = bb[i][0]
							text_y = bb[i][1] + 20
							print('Result Indices: ', best_class_indices[0])
							for H_i in HumanNames:
								if HumanNames[best_class_indices[0]] == H_i:
									predict_names = HumanNames[best_class_indices[0]]
									cv2.putText(frame, predict_names, (text_x, text_y), cv2.FONT_HERSEY_COMPLEX_SMALL, 1, (255, 0, 0) thickness=1, lineType=2)
					else:
						print('Unable to find face')
					cv2.imshow('%d Face(s) detected' % nrof_faces, cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
					if cv2.waitKey(10000) & 0xFF == ord('q'):
						print('Processing next image...')
					elif cv2.waitKey(10000) & 0xFF == ord('q'):
						sys.exit('Ending...')
			cv2.destroyAllWindows()