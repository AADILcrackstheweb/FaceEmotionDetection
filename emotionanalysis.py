from fer import FER
import matplotlib.pyplot as plt 
test_image_one = plt.imread("Image-One.jpeg")
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions(test_image_one)
# Print all captured emotions with the image
print(captured_emotions)
#plt.imshow(test_image_one)
#Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)
# Testing on another image
test_image_three = plt.imread("Image-Two.jpg")
captured_emotions_three = emo_detector.detect_emotions(test_image_three)
print(captured_emotions_three)
plt.imshow(test_image_three)
dominant_emotion_three, emotion_score_three = emo_detector.top_emotion(test_image_three)
print(dominant_emotion_three, emotion_score_three)
# Testing on another image
test_image_four = plt.imread("Image-Three.jpg")
captured_emotions_four = emo_detector.detect_emotions(test_image_four)
print(captured_emotions_four)
plt.imshow(test_image_four)
dominant_emotion_four, emotion_score_four = emo_detector.top_emotion(test_image_four)
print(dominant_emotion_four, emotion_score_four)
# Testing on another image
test_image_five = plt.imread("Image-Four.jpg")
captured_emotions_five = emo_detector.detect_emotions(test_image_five)
print(captured_emotions_five)
plt.imshow(test_image_five)
dominant_emotion_five, emotion_score_five = emo_detector.top_emotion(test_image_five)
print(dominant_emotion_five, emotion_score_five)


