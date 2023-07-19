import cv2
#pip install opencv-contrib-python
#DERİN SİNİR AĞLARI TANIMLAMA
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
#model tanımlama
model = cv2.dnn_DetectionModel(net)
#model ölçeklendirme
model.setInputParams(size=(320,320),scale =1/255)

#sınıf listesi yükleme

classes = []
with open("dnn_model\classes.txt", "r") as file_object:    #r read demek
   for class_name in file_object.readlines():
       class_name = class_name.strip()
       classes.append(class_name)

print("obje listesi")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while True:
   #görüntü alma
   ret, frame = cap.read()
  #class_ids: Bu, tespit edilen nesnelerin sınıf kimliklerini (class IDs) içeren bir dizi veya liste. Her bir tespit için, sınıf kimliği, nesnenin ne tür bir sınıfa ait olduğunu belirtir. Örneğin, bir görüntüdeki arabaları tespit ediyorsanız, sınıf kimliği 1 olabilirken, insanları tespit etmek için sınıf kimliği 2 olabilir. Sınıf kimlikleri genellikle bir sınıflandırma modeline dayanan nesne tespiti modellerinde kullanılır.
  #score: Bu, her tespitin güven skorlarını içeren bir dizi veya liste. Güven skoru, bir tespitin ne kadar doğru olduğunu veya ne kadar güvenilir olduğunu belirten bir değerdir. Genellikle 0 ile 1 arasında bir değerdir, 1'e yaklaştıkça daha güvenilir bir tespit olduğunu gösterir. Yüksek skorlar, güvenilir tespitleri ifade ederken, düşük skorlar daha şüpheli veya yanlış tespitleri gösterebilir.
  #bboxes: Bu, tespit edilen nesnelerin sınırlayıcı kutularını (bounding boxes) içeren bir dizi veya liste. Her bir sınırlayıcı kutu, tespit edilen nesnenin konumunu ve boyutunu belirtir. Bir sınırlayıcı kutu genellikle sol üst köşesinin koordinatlarını (x, y) ve genişlik ile yükseklik değerlerini içerir. Bu kutular, nesnelerin görüntü üzerinde nasıl yerleştirildiğini belirlemek için kullanılır. 
   (class_ids, score, bboxes)= model.detect(frame)
   for class_id, score, bbox in zip(class_ids, score, bboxes) :
      x,y,w,h = bbox
      print(x,y,w,h)

      class_name = classes[class_id]
      cv2.putText(frame,class_name,(x,y-10), cv2.FONT_HERSHEY_PLAIN,2,(200,0,50),2) #bilerek -10 yazdık ki üst üste binmesin 
      cv2.rectangle(frame, (x,y),(x+w,y+h),(200,0,50),2)  #koordinat,en-boy,renk,kalınlık
      

   print("class ids :" , class_ids)
   print("score :", score)
   print("bboxes :", bboxes)

   cv2.imshow("frame", frame)
   cv2.waitKey(1)
   if cv2.waitKey == 27 & 0xFF == ord('q') :
      break


cam.release()
cv2.destroyAllWindows()

