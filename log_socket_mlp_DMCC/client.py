# client  
  
import socket  
    
address = ('127.0.0.1', 31501) 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
s.connect(address)  

# send the file path
s.send("data/daht_c001_04_15.csv")  

data = s.recv(512)  
print 'the data received is '
print data 
        
s.close()  

