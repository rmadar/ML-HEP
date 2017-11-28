from rootpy.io import root_open

f=root_open('flatNtuple_4topSM.root')
t=f.Get('FlatTree')
t.csv()
