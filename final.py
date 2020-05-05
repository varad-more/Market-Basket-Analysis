from flask import Flask, request
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from flask import render_template

import pandas as pd
import numpy as np
import warnings


from matplotlib.figure import Figure
from werkzeug import secure_filename



app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/test')
def chartTest():
  price = (123,123,124)
  lnprice=np.log(price)
  plt.plot(lnprice)   
  plt.savefig('static/new_plot1.png')
  name = 'new_plot'
  url ='new_plot1.png'
  return render_template('bil.html', name = 'new_plot', url ='new_plot1.png')

@app.route('/upload')
def upload_files():
  print("##")
  return render_template('uploader.html')


# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#     #  f.save(secure_filename(f.filename))
#       print(f) ax=plt.subplots(figsize=(16,7))
  # df['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)
  # plt.xlabel('Food Item',fontsize=20)
  # plt.ylabel('Number of transactions',fontsize=17)
  # ax.tick_params(labelsize=20)
  # plt.title('20 Most Sold Items',fontsize=20)
#       return 'file uploaded successfully'
  

@app.route('/final', methods = ['GET', 'POST'])
def bil_aprori():
  if request.method == 'POST':
      f = request.files['file']
    #  f.save(secure_filename(f.filename))
      print(f)
      app.logger.info("File Received")
  else: 
    return 'Error in Upload'

  warnings.filterwarnings('ignore')
  df = pd.read_csv(f)
  print ("Dataset Import Success")
  df['Item']=df['Item'].str.lower()

  x=df['Item']== 'none'
  print(x.value_counts())
  df=df.drop(df[df.Item == 'none'].index)

  len(df['Item'].unique())

  df_for_top10_Items=df['Item'].value_counts().head(10)
  Item_array= np.arange(len(df_for_top10_Items))

  import matplotlib.pyplot as plt
  # plt.figure(figsize=(15,5))
  # Items_name=['coffee','bread','tea','cake','pastry','sandwich','medialuna','hot chocolate','cookies','brownie']
  # plt.bar(Item_array,df_for_top10_Items.iloc[:])
  # plt.xticks(Item_array,Items_name)
  # plt.title('Top 5 most selling items')
  # # plt.show() 
  # plt.savefig('static/new_plot1.png')

  fig, ax=plt.subplots(figsize=(16,7))
  df['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)
  plt.xlabel('Food Item',fontsize=20)
  plt.ylabel('Number of transactions',fontsize=17)
  ax.tick_params(labelsize=20)
  plt.title('20 Most Sold Items',fontsize=20)
  # plt.grid()
  
  plt.savefig('static/new_plot1.png')
  plt.clf()
  plt.cla()
  plt.close()
  ######################################################################

  df['Date'] = pd.to_datetime(df['Date'])
  df['Time'] = pd.to_datetime(df['Time'],format= '%H:%M:%S' ).dt.hour
  df['day_of_week'] = df['Date'].dt.weekday
  d=df.loc[:,'Date']

  weekday_names=[ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  Weekday_number=[0,1,2,3,4,5,6]
  week_df = d.groupby(d.dt.weekday).count().reindex(Weekday_number)
  Item_array_week= np.arange(len(week_df))

  plt.figure(figsize=(15,5))
  my_colors = 'rk'
  plt.bar(Item_array_week,week_df, color=my_colors)
  plt.xticks(Item_array_week,weekday_names)
  plt.title('Number of Transactions made based on Weekdays')
  #plt.show()
  plt.savefig('static/new_plot2.png')

  plt.clf()
  plt.cla()
  plt.close()
  #####################################################################

  dt=df.loc[:,'Time']
  Hour_names=[ 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
  time_df=dt.groupby(dt).count().reindex(Hour_names)
  Item_array_hour= np.arange(len(time_df))

  plt.figure(figsize=(15,5))
  my_colors = 'rb'
  plt.bar(Item_array_hour,time_df, color=my_colors)
  plt.xticks(Item_array_hour,Hour_names)
  plt.title('Number of Transactions made based on Hours')
  #plt.show()
  plt.savefig('static/new_plot3.png')
  plt.clf()
  plt.cla()
  plt.close()

  ##############################################################################

  from mlxtend.frequent_patterns import apriori
  from mlxtend.frequent_patterns import association_rules

  hot_encoded_df=df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
  def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
  hot_encoded_df = hot_encoded_df.applymap(encode_units)

  frequent_itemsets = apriori(hot_encoded_df, min_support=0.01, use_colnames=True)

  rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
  rules.head()
  rules[ (rules['lift'] >= 1) &
       (rules['confidence'] >= 0.5) ]

  ###############################################################################
  support=rules['support'].values
  confidence=rules['confidence'].values
  import seaborn as sns
 
  for i in range (len(support)):
    support[i] = support[i] 
    confidence[i] = confidence[i]
    

  # plt.figure(figsize=(15,5))
  # my_colors = 'rb'
  # plt.bar(Item_array_hour,time_df, color=my_colors)
  # plt.xticks(Item_array_hour,Hour_names)
  # plt.title('Number of Transactions made based on Hours')
  # #plt.show()
  # plt.savefig('static/new_plot3.png')



  plt.plot()
  plt.figure(figsize=(15,5))
  plt.scatter(support, confidence,   alpha=0.5, marker="*")
  plt.title('Association Rules')
  plt.xlabel('support')
  plt.ylabel('confidence')    
  #fig=()
  #sns.regplot(x=support, y=confidence, fit_reg=False)
  # plt.show(p)
  # fig = p.get_figure()
  # fig.savefig('out.png')
  # fig = fig1.get_figure()
  # fig.savefig("output.png")
  # fig = sns.regplot(x=support, y=confidence, fit_reg=False)
  # fig.figure.savefig('static/new_plot4.png')

  #fig = sns.regplot(x=support, y=confidence, fit_reg=False)
  #fig.figure.savefig('../test.png')

  #plt.show()
  plt.savefig('static/new_plot4.png')

  plt.clf()
  plt.cla()
  plt.close()
  


  ######################
  rules_to_show =20
  import networkx as nx  
  plt.plot()
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([c])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  
  nx.draw_networkx_labels(G1, pos)
  # plt.show()
  plt.savefig('static/new_plot5.png')
  plt.clf()
  plt.cla()
  plt.close()

  import time
  time.sleep(5)   # Delays for 5 seconds. You can also use a float value.
  return render_template('out.html', name = 'Top 5 most selling items', url ='new_plot1.png', name1 ='abc' , url2='new_plot2.png', url3='new_plot3.png', url4='new_plot4.png', url5='new_plot5.png')



