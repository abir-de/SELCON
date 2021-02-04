

ts_size = {'cadata':0, 'LawSchool':0, 'MSD': 0, 'NY_Stock_exchange_close': 835856,
          'NY_Stock_exchange_high':835856,'Comm_Crime':199}

files = ['results/Faster/NY_Stock_exchange_high_100/0.001/1.0/35/NY_Stock_exchange_high_model.txt']

wb = open(files[0])

line = wb.readline()
while line:
    print(line[:10],line[-10:])
    
    line = wb.readline()