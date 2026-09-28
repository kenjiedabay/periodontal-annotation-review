import csv,tempfile,unittest
from pathlib import Path
from PIL import Image
from app.brar_review_pilot import create_package
class BrarPilotTest(unittest.TestCase):
 def test_balanced_blinded_package(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'BRAR Dataset'/'BRAR-anchored multimodal dataset'; root.mkdir(parents=True); fields=['File name','Age','Gender','Bone resorption','Bone resorption Age','Level','Number of missing teeth','Implant','Residual root','Functional tooth logarithm']
   with (root/'meta_data.csv').open('w',newline='',encoding='utf-8') as h:
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
    for level in (1,2,3):
     folder=root/f'level_{level}'; folder.mkdir()
     for i in range(5):
      name=f'{level}_{i}.jpg'; Image.new('L',(20,10),level*20+i).save(folder/name); w.writerow({'File name':name,'Age':20,'Gender':0,'Bone resorption':.2,'Bone resorption Age':level/2,'Level':level,'Number of missing teeth':0,'Implant':0,'Residual root':0,'Functional tooth logarithm':14})
   package=create_package(root,Path(temp)/'out',12,4); self.assertEqual([sum(e['published_level']==x for e in package['entries']) for x in (1,2,3)],[4,4,4]); self.assertNotIn('published_level',(Path(temp)/'out'/'pilot_manifest.public.json').read_text())
if __name__=='__main__': unittest.main()
