import csi_utils.pipeline_utils as pipeline_utils
import tqdm
import numpy as np

def generate_static_compensation(bag, tx, transpose=False):
  channels = {}
  times = {}
  rssis = {}
  rx = None
  for topic, msg, t in tqdm.tqdm(bag.read_messages('/csi')):
      csi = pipeline_utils.extract_csi(msg)
      #assuming this does not change
      rx = msg.rx_id
      mac = pipeline_utils.mac_to_str(tuple(msg.txmac))

      key = f"{mac}-{msg.chan}-{msg.bw}"

      if key not in channels.keys():
          channels[key] = []
          times[key] = []
          rssis[key] = []
      if transpose:
        csi = csi.transpose(0,2,1)
      channels[key].append(csi)
      times[key].append(t.to_sec())
      rssis[key].append(msg.rssi)

  maxk = ''
  maxlen = 0
  for key in channels:
    if len(channels[key]) > maxlen:
        maxk = key
        maxlen = len(channels[key])
  if maxk == '':
    print("No CSI in bag file")
    exit(1)

  ksplit = maxk.split("-")
  print(f"Generating Compensation for {rx} with chanspec {ksplit[1]}/{ksplit[2]} from txmac {ksplit[0]}")
  Hcomp = np.asarray(channels[maxk])[:,:,:,tx].transpose(1,2,0)
  #reduce to singular vector
  nsub = Hcomp.shape[0]
  nrx = Hcomp.shape[1]
  Hcomp = Hcomp.reshape((-1,Hcomp.shape[2]))
  _,vec=np.linalg.eig(Hcomp@Hcomp.conj().T)
  Hcomp = vec[:,0].reshape((nsub,nrx,1))
  Hcomp = np.exp(-1.0j*np.angle(Hcomp))
  return Hcomp
