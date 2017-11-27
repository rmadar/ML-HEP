from rootpy.io import root_open

f=root_open('flatNtuple_ttV.root')
t=f.Get('FlatTree')
t.csv()
