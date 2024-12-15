import cv2, time, pandas
from datetime import datetime

first_frame= None

status_list= [None, None] # trick for no errors with [-2]
times=[]

video= cv2.VideoCapture(0)

while True:
    check, frame= video.read()

    status= 0

    frame_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray= cv2.GaussianBlur(frame_gray, (21,21),0) # for better precision and lower noise (standard numbers)

    if first_frame is None:
        first_frame= frame_gray
        continue # this restarts the loop


    delta_frame= cv2.absdiff(first_frame, frame_gray)
    thres_frame= cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1] # if >=30 turns to 255
    #with binary method only need 2nd value
    thres_frame= cv2.dilate(thres_frame, None, iterations=2) #bigger number bigger smooth


    (cnts,_)= cv2.findContours(thres_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #store contours in a tuple                           method              aproximation method
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue # if less 10000 area restarts the loop
        
        
        status=1 # if contour valid --> 1
        

        (x, y, w, h)= cv2.boundingRect(contour)

        if w==640 and h==480: # no rectangle if burned image
            continue
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    
    status_list.append(status)

    if status_list[-1]==1 and status_list[-2]==0: # movements enter
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1: # movements exit
        times.append(datetime.now())


    cv2.imshow("Gray_frames", frame_gray)
    cv2.imshow("Delta_frames", delta_frame)
    cv2.imshow("Threshold_frames", thres_frame)
    cv2.imshow("Frame", frame)


    key= cv2.waitKey(1)

    if key==ord("q"): # if press key q shut down the program

        if status==1:
            times.append(datetime.now())
        
        break
    

# create file with data detections
start=[]
end=[]

for i in range(0,len(times),2):
    start.append(times[i])
    end.append(times[i+1])


df= pandas.DataFrame({"START":start,
     "END":end})

#print(df)

df.to_csv("Times.csv")
    

video.release()
cv2.destroyAllWindows()