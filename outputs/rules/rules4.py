def findDecision(obj): #obj[0]: pts_per_g, obj[1]: per, obj[2]: ws, obj[3]: ws_per_48, obj[4]: bpm
   if obj[1] == '22.sep':
      if obj[4] == '4.maj':
         return 0.252
      elif obj[4] == '6.avg':
         return 0.035
      elif obj[4] == '3.feb':
         return 0.003
      elif obj[4] == '4.sep':
         return 0.264
      elif obj[4] == '4.avg':
         return 0.033
      elif obj[4] == '3.jul':
         return 0.031
      elif obj[4] == '4.feb':
         return 0.001
      elif obj[4] == '6.mar':
         return 0.017
      elif obj[4] == '2.jun':
         return 0.045
      elif obj[4] == '1.jun':
         return 0.12
      elif obj[4] == '8.mar':
         return 0.09699999999999999
      elif obj[4] == '2.feb':
         return 0.048
      elif obj[4] == '5.mar':
         return 0.002
      elif obj[4] == '5.jun':
         return 0.006
      elif obj[4] == '5.sep':
         return 0.129
      elif obj[4] == '6.maj':
         return 0.017
      elif obj[4] == '4.jun':
         return 0.043
      else:
         return 0.043
   elif obj[1] == '21.sep':
      if obj[0] == '23.sep':
         return 0.21899999999999997
      elif obj[0] == '24.mar':
         return 0.005
      elif obj[0] == '26.maj':
         return 0.004
      elif obj[0] == '25.jan':
         return 0.001
      elif obj[0] == '17.maj':
         return 0.001
      elif obj[0] == '22.maj':
         return 0.004
      elif obj[0] == '27.0':
         return 0.48700000000000004
      elif obj[0] == '21.jan':
         return 0.001
      elif obj[0] == '31.4':
         return 0.023
      elif obj[0] == '27.sep':
         return 0.29100000000000004
      elif obj[0] == '14.jul':
         return 0.011000000000000001
      else:
         return 0.011000000000000001
   elif obj[1] == '22.jun':
      if obj[0] == '23.jan':
         return 0.004
      elif obj[0] == '21.jan':
         return 0.128
      elif obj[0] == '17.jul':
         return 0.001
      elif obj[0] == '22.sep':
         return 0.588
      elif obj[0] == '22.maj':
         return 0.301
      elif obj[0] == '21.apr':
         return 0.079
      elif obj[0] == '17.jun':
         return 0.401
      elif obj[0] == '19.jun':
         return 0.065
      elif obj[0] == '22.jun':
         return 0.033
      elif obj[0] == '16.maj':
         return 0.002
      elif obj[0] == '19.sep':
         return 0.19
      else:
         return 0.19
   elif obj[1] == '23.jan':
      if obj[0] == '20.apr':
         return 0.002
      elif obj[0] == '25.sep':
         return 0.002
      elif obj[0] == '24.jan':
         return 0.034
      elif obj[0] == '22.sep':
         return 0.004
      elif obj[0] == '21.apr':
         return 0.149
      elif obj[0] == '18.sep':
         return 0.003
      elif obj[0] == '18.jun':
         return 0.026000000000000002
      elif obj[0] == '21.jul':
         return 0.03
      elif obj[0] == '19.jun':
         return 0.635
      else:
         return 0.635
   elif obj[1] == '18.0':
      if obj[0] == '13.feb':
         return 0.011000000000000001
      elif obj[0] == '20.apr':
         return 0.008
      elif obj[0] == '21.jun':
         return 0.001
      elif obj[0] == '22.jun':
         return 0.027999999999999997
      elif obj[0] == '10.mar':
         return 0.001
      elif obj[0] == '21.jul':
         return 0.004
      elif obj[0] == '18.jul':
         return 0.016
      elif obj[0] == '16.maj':
         return 0.004
      elif obj[0] == '23.maj':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '19.jul':
      if obj[0] == '17.feb':
         return 0.001
      elif obj[0] == '22.sep':
         return 0.001
      elif obj[0] == '22.maj':
         return 0.003
      elif obj[0] == '15.maj':
         return 0.013999999999999999
      elif obj[0] == '26.avg':
         return 0.11699999999999999
      elif obj[0] == '14.mar':
         return 0.006
      elif obj[0] == '19.apr':
         return 0.026000000000000002
      elif obj[0] == '19.mar':
         return 0.001
      elif obj[0] == '18.apr':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '22.jul':
      if obj[4] == '4.feb':
         return 0.001
      elif obj[4] == '4.jun':
         return 0.006999999999999999
      elif obj[4] == '4.maj':
         return 0.17300000000000001
      elif obj[4] == '4.apr':
         return 0.005
      elif obj[4] == '3.avg':
         return 0.069
      elif obj[4] == '0.7':
         return 0.006999999999999999
      elif obj[4] == '3.feb':
         return 0.083
      elif obj[4] == '3.jun':
         return 0.491
      elif obj[4] == '3.0':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '19.sep':
      if obj[0] == '21.jul':
         return 0.001
      elif obj[0] == '19.jul':
         return 0.002
      elif obj[0] == '20.0':
         return 0.001
      elif obj[0] == '17.feb':
         return 0.004
      elif obj[0] == '21.feb':
         return 0.613
      elif obj[0] == '16.jul':
         return 0.002
      elif obj[0] == '25.jun':
         return 0.084
      elif obj[0] == '21.sep':
         return 0.017
      else:
         return 0.017
   elif obj[1] == '23.apr':
      if obj[0] == '23.sep':
         return 0.045
      elif obj[0] == '25.jul':
         return 0.042
      elif obj[0] == '17.feb':
         return 0.016
      elif obj[0] == '18.maj':
         return 0.344
      elif obj[0] == '22.avg':
         return 0.05
      elif obj[0] == '24.maj':
         return 0.079
      elif obj[0] == '22.0':
         return 0.006999999999999999
      elif obj[0] == '23.0':
         return 0.09300000000000001
      else:
         return 0.09300000000000001
   elif obj[1] == '21.mar':
      if obj[0] == '20.jun':
         return 0.032
      elif obj[0] == '25.maj':
         return 0.003
      elif obj[0] == '22.sep':
         return 0.002
      elif obj[0] == '21.maj':
         return 0.201
      elif obj[0] == '20.feb':
         return 0.005
      elif obj[0] == '17.0':
         return 0.003
      elif obj[0] == '15.jan':
         return 0.001
      elif obj[0] == '22.apr':
         return 0.05
      else:
         return 0.05
   elif obj[1] == '21.jul':
      if obj[0] == '17.apr':
         return 0.017
      elif obj[0] == '21.jun':
         return 0.003
      elif obj[0] == '22.maj':
         return 0.002
      elif obj[0] == '17.sep':
         return 0.006
      elif obj[0] == '17.mar':
         return 0.002
      elif obj[0] == '17.0':
         return 0.026000000000000002
      elif obj[0] == '19.apr':
         return 0.001
      elif obj[0] == '29.avg':
         return 0.071
      else:
         return 0.071
   elif obj[1] == '20.apr':
      if obj[0] == '17.avg':
         return 0.001
      elif obj[0] == '23.feb':
         return 0.003
      elif obj[0] == '17.maj':
         return 0.002
      elif obj[0] == '18.maj':
         return 0.004
      elif obj[0] == '12.jan':
         return 0.003
      elif obj[0] == '19.jan':
         return 0.012
      elif obj[0] == '19.feb':
         return 0.021
      elif obj[0] == '20.0':
         return 0.009000000000000001
      else:
         return 0.009000000000000001
   elif obj[1] == '21.avg':
      if obj[0] == '26.jul':
         return 0.001
      elif obj[0] == '23.apr':
         return 0.16699999999999998
      elif obj[0] == '23.mar':
         return 0.027999999999999997
      elif obj[0] == '22.jul':
         return 0.046
      elif obj[0] == '24.feb':
         return 0.228
      elif obj[0] == '12.0':
         return 0.004
      elif obj[0] == '29.avg':
         return 0.002
      elif obj[0] == '21.avg':
         return 0.091
      else:
         return 0.091
   elif obj[1] == '23.jun':
      if obj[2] == '6.mar':
         return 0.006
      elif obj[2] == '10.sep':
         return 0.02
      elif obj[2] == '14.jun':
         return 0.001
      elif obj[2] == '13.sep':
         return 0.149
      elif obj[2] == '12.0':
         return 0.157
      elif obj[2] == '12.avg':
         return 0.584
      elif obj[2] == '13.jun':
         return 0.001
      elif obj[2] == '11.jun':
         return 0.337
      else:
         return 0.337
   elif obj[1] == '22.jan':
      if obj[0] == '25.jul':
         return 0.001
      elif obj[0] == '21.maj':
         return 0.078
      elif obj[0] == '23.jun':
         return 0.177
      elif obj[0] == '22.jul':
         return 0.235
      elif obj[0] == '14.apr':
         return 0.003
      elif obj[0] == '28.sep':
         return 0.001
      elif obj[0] == '26.sep':
         return 0.011000000000000001
      elif obj[0] == '18.feb':
         return 0.01
      else:
         return 0.01
   elif obj[1] == '24.feb':
      if obj[0] == '20.jun':
         return 0.011000000000000001
      elif obj[0] == '32.3':
         return 0.159
      elif obj[0] == '23.apr':
         return 0.034
      elif obj[0] == '24.feb':
         return 0.858
      elif obj[0] == '30.mar':
         return 0.01
      elif obj[0] == '28.mar':
         return 0.873
      elif obj[0] == '23.maj':
         return 0.247
      else:
         return 0.247
   elif obj[1] == '19.jun':
      if obj[0] == '22.feb':
         return 0.05
      elif obj[0] == '17.avg':
         return 0.001
      elif obj[0] == '18.mar':
         return 0.001
      elif obj[0] == '13.avg':
         return 0.001
      elif obj[0] == '19.mar':
         return 0.087
      elif obj[0] == '18.feb':
         return 0.017
      elif obj[0] == '19.jun':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '22.feb':
      if obj[0] == '23.mar':
         return 0.003
      elif obj[0] == '28.feb':
         return 0.053
      elif obj[0] == '13.avg':
         return 0.002
      elif obj[0] == '21.feb':
         return 0.081
      elif obj[0] == '25.apr':
         return 0.017
      elif obj[0] == '26.avg':
         return 0.27
      elif obj[0] == '18.jul':
         return 0.026000000000000002
      else:
         return 0.026000000000000002
   elif obj[1] == '20.jun':
      if obj[2] == '11.mar':
         return 0.021
      elif obj[2] == '12.jun':
         return 0.048
      elif obj[2] == '8.jul':
         return 0.001
      elif obj[2] == '10.mar':
         return 0.006
      elif obj[2] == '10.0':
         return 0.01
      elif obj[2] == '11.jul':
         return 0.061
      elif obj[2] == '8.sep':
         return 0.027000000000000003
      else:
         return 0.027000000000000003
   elif obj[1] == '23.feb':
      if obj[0] == '30.jul':
         return 0.18899999999999997
      elif obj[0] == '24.jun':
         return 0.085
      elif obj[0] == '18.mar':
         return 0.33799999999999997
      elif obj[0] == '25.feb':
         return 0.078
      elif obj[0] == '22.0':
         return 0.386
      elif obj[0] == '21.jul':
         return 0.627
      elif obj[0] == '14.jul':
         return 0.019
      else:
         return 0.019
   elif obj[1] == '24.jan':
      if obj[0] == '28.apr':
         return 0.055999999999999994
      elif obj[0] == '24.mar':
         return 0.07
      elif obj[0] == '23.jun':
         return 0.485
      elif obj[0] == '23.apr':
         return 0.025
      elif obj[0] == '20.sep':
         return 0.005
      elif obj[0] == '22.apr':
         return 0.005
      elif obj[0] == '27.0':
         return 0.10800000000000001
      else:
         return 0.10800000000000001
   elif obj[1] == '25.jan':
      if obj[0] == '26.jul':
         return 0.507
      elif obj[0] == '24.jun':
         return 0.6579999999999999
      elif obj[0] == '24.maj':
         return 0.96
      elif obj[0] == '25.jun':
         return 0.31
      elif obj[0] == '27.avg':
         return 0.261
      elif obj[0] == '19.apr':
         return 0.518
      elif obj[0] == '23.0':
         return 0.145
      else:
         return 0.145
   elif obj[1] == '20.sep':
      if obj[0] == '23.sep':
         return 0.032
      elif obj[0] == '22.sep':
         return 0.001
      elif obj[0] == '27.apr':
         return 0.004
      elif obj[0] == '20.feb':
         return 0.055999999999999994
      elif obj[0] == '27.feb':
         return 0.013000000000000001
      elif obj[0] == '21.mar':
         return 0.151
      elif obj[0] == '23.0':
         return 0.016
      else:
         return 0.016
   elif obj[1] == '23.0':
      if obj[0] == '25.sep':
         return 0.027000000000000003
      elif obj[0] == '21.maj':
         return 0.025
      elif obj[0] == '27.mar':
         return 0.152
      elif obj[0] == '20.mar':
         return 0.071
      elif obj[0] == '16.avg':
         return 0.406
      elif obj[0] == '21.avg':
         return 0.004
      elif obj[0] == '19.feb':
         return 0.002
      else:
         return 0.002
   elif obj[1] == '24.0':
      if obj[0] == '28.apr':
         return 0.024
      elif obj[0] == '31.jan':
         return 0.904
      elif obj[0] == '22.jun':
         return 0.006
      elif obj[0] == '18.mar':
         return 0.389
      elif obj[0] == '21.feb':
         return 0.004
      elif obj[0] == '18.avg':
         return 0.263
      elif obj[0] == '26.jan':
         return 0.326
      else:
         return 0.326
   elif obj[1] == '21.jun':
      if obj[0] == '25.sep':
         return 0.003
      elif obj[0] == '19.avg':
         return 0.004
      elif obj[0] == '27.sep':
         return 0.015
      elif obj[0] == '21.mar':
         return 0.004
      elif obj[0] == '19.feb':
         return 0.37200000000000005
      elif obj[0] == '18.jun':
         return 0.005
      elif obj[0] == '16.maj':
         return 0.04
      else:
         return 0.04
   elif obj[1] == '24.apr':
      if obj[2] == '8.jun':
         return 0.028999999999999998
      elif obj[2] == '11.jan':
         return 0.02
      elif obj[2] == '12.jul':
         return 0.5770000000000001
      elif obj[2] == '10.jan':
         return 0.002
      elif obj[2] == '9.sep':
         return 0.145
      elif obj[2] == '8.mar':
         return 0.054000000000000006
      elif obj[2] == '15.feb':
         return 0.426
      else:
         return 0.426
   elif obj[1] == '21.jan':
      if obj[0] == '20.jun':
         return 0.015
      elif obj[0] == '25.jul':
         return 0.002
      elif obj[0] == '20.jan':
         return 0.425
      elif obj[0] == '16.sep':
         return 0.013999999999999999
      elif obj[0] == '21.apr':
         return 0.002
      elif obj[0] == '14.jun':
         return 0.006999999999999999
      else:
         return 0.006999999999999999
   elif obj[1] == '18.apr':
      if obj[0] == '18.0':
         return 0.004
      elif obj[0] == '18.sep':
         return 0.009000000000000001
      elif obj[0] == '21.jun':
         return 0.015
      elif obj[0] == '14.mar':
         return 0.021
      elif obj[0] == '18.avg':
         return 0.001
      elif obj[0] == '21.feb':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '23.jul':
      if obj[0] == '30.jul':
         return 0.069
      elif obj[0] == '14.apr':
         return 0.006
      elif obj[0] == '22.feb':
         return 0.033
      elif obj[0] == '24.0':
         return 0.172
      elif obj[0] == '15.sep':
         return 0.002
      elif obj[0] == '26.jun':
         return 0.021
      else:
         return 0.021
   elif obj[1] == '22.maj':
      if obj[0] == '24.jun':
         return 0.27899999999999997
      elif obj[0] == '21.avg':
         return 0.003
      elif obj[0] == '15.jan':
         return 0.001
      elif obj[0] == '16.sep':
         return 0.135
      elif obj[0] == '15.apr':
         return 0.002
      elif obj[0] == '16.maj':
         return 0.002
      else:
         return 0.002
   elif obj[1] == '27.0':
      if obj[0] == '23.sep':
         return 0.94
      elif obj[0] == '25.maj':
         return 0.757
      elif obj[0] == '19.avg':
         return 0.318
      elif obj[0] == '22.sep':
         return 0.813
      elif obj[0] == '20.mar':
         return 0.258
      elif obj[0] == '25.avg':
         return 0.111
      else:
         return 0.111
   elif obj[1] == '21.feb':
      if obj[0] == '20.feb':
         return 0.004
      elif obj[0] == '21.jan':
         return 0.02
      elif obj[0] == '27.jun':
         return 0.07
      elif obj[0] == '18.0':
         return 0.044000000000000004
      elif obj[0] == '20.sep':
         return 0.036000000000000004
      elif obj[0] == '18.apr':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '25.feb':
      if obj[0] == '19.mar':
         return 0.003
      elif obj[0] == '18.sep':
         return 0.077
      elif obj[0] == '28.jul':
         return 0.934
      elif obj[0] == '32.9':
         return 0.09
      elif obj[0] == '23.0':
         return 0.091
      elif obj[0] == '24.avg':
         return 0.21100000000000002
      else:
         return 0.21100000000000002
   elif obj[1] == '20.avg':
      if obj[2] == '9.jun':
         return 0.006999999999999999
      elif obj[2] == '9.jul':
         return 0.013999999999999999
      elif obj[2] == '12.sep':
         return 0.207
      elif obj[2] == '10.jun':
         return 0.366
      elif obj[2] == '11.mar':
         return 0.003
      elif obj[2] == '10.jul':
         return 0.022000000000000002
      else:
         return 0.022000000000000002
   elif obj[1] == '19.maj':
      if obj[0] == '17.mar':
         return 0.001
      elif obj[0] == '14.jan':
         return 0.001
      elif obj[0] == '16.feb':
         return 0.001
      elif obj[0] == '20.0':
         return 0.043
      elif obj[0] == '18.jun':
         return 0.02
      elif obj[0] == '17.0':
         return 0.004
      else:
         return 0.004
   elif obj[1] == '19.feb':
      if obj[0] == '23.jan':
         return 0.003
      elif obj[0] == '17.mar':
         return 0.012
      elif obj[0] == '20.avg':
         return 0.001
      elif obj[0] == '22.jun':
         return 0.001
      elif obj[0] == '14.avg':
         return 0.001
      elif obj[0] == '18.maj':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '23.sep':
      if obj[0] == '21.jan':
         return 0.033
      elif obj[0] == '22.0':
         return 0.122
      elif obj[0] == '17.feb':
         return 0.01
      elif obj[0] == '23.feb':
         return 0.006999999999999999
      elif obj[0] == '25.mar':
         return 0.354
      else:
         return 0.354
   elif obj[1] == '23.avg':
      if obj[0] == '21.feb':
         return 0.013000000000000001
      elif obj[0] == '22.feb':
         return 0.569
      elif obj[0] == '23.apr':
         return 0.036000000000000004
      elif obj[0] == '18.jun':
         return 0.785
      elif obj[0] == '24.jan':
         return 0.012
      else:
         return 0.012
   elif obj[1] == '17.apr':
      if obj[0] == '16.feb':
         return 0.011000000000000001
      elif obj[0] == '22.jan':
         return 0.033
      elif obj[0] == '22.mar':
         return 0.006
      elif obj[0] == '16.jan':
         return 0.001
      elif obj[0] == '20.jan':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '20.maj':
      if obj[0] == '20.apr':
         return 0.026000000000000002
      elif obj[0] == '18.sep':
         return 0.033
      elif obj[0] == '28.jun':
         return 0.003
      elif obj[0] == '26.feb':
         return 0.04
      elif obj[0] == '20.0':
         return 0.004
      else:
         return 0.004
   elif obj[1] == '22.avg':
      if obj[0] == '24.0':
         return 0.10400000000000001
      elif obj[0] == '19.avg':
         return 0.003
      elif obj[0] == '21.jun':
         return 0.005
      elif obj[0] == '23.jul':
         return 0.013000000000000001
      elif obj[0] == '15.avg':
         return 0.019
      else:
         return 0.019
   elif obj[1] == '19.avg':
      if obj[0] == '25.0':
         return 0.005
      elif obj[0] == '23.feb':
         return 0.013999999999999999
      elif obj[0] == '19.maj':
         return 0.002
      elif obj[0] == '20.sep':
         return 0.092
      elif obj[0] == '20.jan':
         return 0.013999999999999999
      else:
         return 0.013999999999999999
   elif obj[1] == '19.apr':
      if obj[0] == '16.sep':
         return 0.015
      elif obj[0] == '20.jul':
         return 0.001
      elif obj[0] == '17.jul':
         return 0.004
      elif obj[0] == '19.apr':
         return 0.016
      elif obj[0] == '19.jan':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '20.jul':
      if obj[2] == '7.maj':
         return 0.003
      elif obj[2] == '9.sep':
         return 0.004
      elif obj[2] == '3.mar':
         return 0.013000000000000001
      elif obj[2] == '10.jan':
         return 0.045
      elif obj[2] == '6.0':
         return 0.001
      else:
         return 0.001
   elif obj[1] == '24.jun':
      if obj[0] == '30.jun':
         return 0.11599999999999999
      elif obj[0] == '20.jan':
         return 0.655
      elif obj[0] == '23.jun':
         return 0.004
      elif obj[0] == '21.jul':
         return 0.319
      elif obj[0] == '29.avg':
         return 0.003
      else:
         return 0.003
   elif obj[1] == '26.apr':
      return 0.3435
   elif obj[1] == '26.feb':
      return 0.41425
   elif obj[1] == '22.0':
      return 0.305
   elif obj[1] == '15.jan':
      return 0.004
   elif obj[1] == '27.jun':
      return 0.27725
   elif obj[1] == '19.0':
      return 0.0037500000000000003
   elif obj[1] == '17.feb':
      return 0.00775
   elif obj[1] == '18.feb':
      return 0.0045000000000000005
   elif obj[1] == '27.jan':
      return 0.37825000000000003
   elif obj[1] == '24.avg':
      return 0.18824999999999997
   elif obj[1] == '23.maj':
      return 0.302
   elif obj[1] == '18.avg':
      return 0.013250000000000001
   elif obj[1] == '24.mar':
      return 0.022
   elif obj[1] == '26.jan':
      return 0.358
   elif obj[1] == '17.mar':
      return 0.011000000000000001
   elif obj[1] == '20.mar':
      return 0.0055
   elif obj[1] == '16.jul':
      return 0.001
   elif obj[1] == '24.maj':
      return 0.06575
   elif obj[1] == '19.mar':
      return 0.006750000000000001
   elif obj[1] == '25.jun':
      return 0.4345
   elif obj[1] == '18.jan':
      return 0.002
   elif obj[1] == '21.0':
      return 0.068
   elif obj[1] == '25.maj':
      return 0.3173333333333333
   elif obj[1] == '22.apr':
      return 0.060666666666666674
   elif obj[1] == '27.avg':
      return 0.5073333333333333
   elif obj[1] == '17.sep':
      return 0.001
   elif obj[1] == '23.mar':
      return 0.43533333333333335
   elif obj[1] == '18.sep':
      return 0.004666666666666667
   elif obj[1] == '28.sep':
      return 0.36333333333333334
   elif obj[1] == '25.sep':
      return 0.38233333333333325
   elif obj[1] == '25.avg':
      return 0.217
   elif obj[1] == '29.apr':
      return 0.8283333333333333
   elif obj[1] == '19.jan':
      return 0.262
   elif obj[1] == '22.mar':
      return 0.007333333333333334
   elif obj[1] == '14.mar':
      return 0.018333333333333333
   elif obj[1] == '17.jan':
      return 0.005
   elif obj[1] == '18.jun':
      return 0.025333333333333333
   elif obj[1] == '14.sep':
      return 0.008666666666666668
   elif obj[1] == '20.jan':
      return 0.07766666666666666
   elif obj[1] == '20.0':
      return 0.06066666666666667
   elif obj[1] == '25.apr':
      return 0.19733333333333336
   elif obj[1] == '17.jun':
      return 0.002
   elif obj[1] == '17.0':
      return 0.0495
   elif obj[1] == '29.jan':
      return 0.603
   elif obj[1] == '26.mar':
      return 0.058
   elif obj[1] == '17.maj':
      return 0.007
   elif obj[1] == '25.0':
      return 0.0035
   elif obj[1] == '28.mar':
      return 0.671
   elif obj[1] == '13.jun':
      return 0.0045000000000000005
   elif obj[1] == '28.jan':
      return 0.49250000000000005
   elif obj[1] == '25.jul':
      return 0.0495
   elif obj[1] == '28.0':
      return 0.2415
   elif obj[1] == '16.apr':
      return 0.012
   elif obj[1] == '30.jul':
      return 0.8055
   elif obj[1] == '26.0':
      return 0.10750000000000001
   elif obj[1] == '17.jul':
      return 0.0105
   elif obj[1] == '31.6':
      return 0.963
   elif obj[1] == '26.jun':
      return 0.3615
   elif obj[1] == '25.mar':
      return 0.706
   elif obj[1] == '29.jul':
      return 0.5645
   elif obj[1] == '30.jun':
      return 0.5365
   elif obj[1] == '15.maj':
      return 0.004
   elif obj[1] == '26.sep':
      return 0.7949999999999999
   elif obj[1] == '21.apr':
      return 0.001
   elif obj[1] == '26.maj':
      return 0.509
   elif obj[1] == '24.sep':
      return 0.0285
   elif obj[1] == '24.jul':
      return 0.2185
   elif obj[1] == '31.jan':
      return 0.8420000000000001
   elif obj[1] == '17.avg':
      return 0.0125
   elif obj[1] == '15.jun':
      return 0.001
   elif obj[1] == '27.mar':
      return 0.5455
   elif obj[1] == '31.jul':
      return 0.9
   elif obj[1] == '18.mar':
      return 0.007000000000000001
   elif obj[1] == '18.jul':
      return 0.001
   elif obj[1] == '28.maj':
      return 0.28600000000000003
   elif obj[1] == '12.feb':
      return 0.004
   elif obj[1] == '29.avg':
      return 0.5760000000000001
   elif obj[1] == '27.apr':
      return 0.496
   elif obj[1] == '30.mar':
      return 0.359
   elif obj[1] == '13.maj':
      return 0.004
   elif obj[1] == '13.sep':
      return 0.001
   elif obj[1] == '15.sep':
      return 0.001
   elif obj[1] == '16.jun':
      return 0.009000000000000001
   elif obj[1] == '14.jul':
      return 0.001
   elif obj[1] == '30.apr':
      return 0.562
   elif obj[1] == '30.2':
      return 0.466
   elif obj[1] == '13.apr':
      return 0.001
   elif obj[1] == '28.avg':
      return 0.268
   elif obj[1] == '16.avg':
      return 0.001
   elif obj[1] == '16.jan':
      return 0.012
   elif obj[1] == '20.feb':
      return 0.001
   elif obj[1] == '16.feb':
      return 0.006
   elif obj[1] == '10.jun':
      return 0.004
   elif obj[1] == '26.avg':
      return 0.735
   elif obj[1] == '27.jul':
      return 0.938
   elif obj[1] == '10.jan':
      return 0.001
   elif obj[1] == '28.jun':
      return 0.5760000000000001
   elif obj[1] == '21.maj':
      return 0.033
   elif obj[1] == '27.feb':
      return 0.233
   elif obj[1] == '16.maj':
      return 0.001
   elif obj[1] == '18.maj':
      return 0.005
   elif obj[1] == '27.sep':
      return 0.726
   elif obj[1] == '28.feb':
      return 0.012
   elif obj[1] == '13.avg':
      return 0.006999999999999999
   elif obj[1] == '31.2':
      return 0.613
   elif obj[1] == '15.jul':
      return 0.003
   elif obj[1] == '14.maj':
      return 0.004
   elif obj[1] == '30.0':
      return 0.159
   elif obj[1] == '29.maj':
      return 0.106
   elif obj[1] == '27.maj':
      return 0.35100000000000003
   else:
      return 0.35100000000000003
