require 'hypero'
conn = hypero.connect { database='junebug', username='junebug' }

batName = "Two-stream Video Classification - UCF101"
verDesc = "minor changes"
battery = conn.battery (batName, verDesc)

hex = bat:experiment()
hp = { startLR = 0.005, momentum = 0.09, lrDecay = 'linear', minLR = 0.0001, satEpoch = 10 }
md = { hostname = 'sunwoo', dataset = 'ucf-101' }
res = { trainAcc = 0.98, validAcc = 0.92, testAcc = 0.91 }

hex:setParam (hp)
hex:setMeta (md)
hex:setResult (res)


