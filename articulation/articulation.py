
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa


Compute articulation features from continuous speech.

122 descriptors are computed:

1-22. Bark band energies in onset transitions (22 BBE).
23-34. Mel frequency cepstral coefficients in onset transitions (12 MFCC onset)
35-46. First derivative of the MFCCs in onset transitions (12 DMFCC onset)
47-58. Second derivative of the MFCCs in onset transitions (12 DDMFCC onset)
59-80. Bark band energies in offset transitions (22 BBE).
81-92. MFCCC in offset transitions (12 MFCC offset)
93-104. First derivative of the MFCCs in offset transitions (12 DMFCC offset)
105-116. Second derivative of the MFCCs in offset transitions (12 DMFCC offset)
117. First formant Frequency
118. First Derivative of the first formant frequency
119. Second Derivative of the first formant frequency
120. Second formant Frequency
121. First derivative of the Second formant Frequency
122. Second derivative of the Second formant Frequency


Static or dynamic matrices can be computed:

Static matrix is formed with 488 features formed with (122 descriptors) x (4 functionals: mean, std, skewness, kurtosis)

For dynamic analysis, two matrices are created for onset and offset based features

Dynamic matrices are formed with the 58 descriptors (22 BBEs, 12 MFCC, 12DMFCC, 12 DDMFCC ) computed for frames of 40 ms.

The first two frames of each recording are not considered for dynamic analysis to be able to stack the derivatives of MFCCs



Notes:
The fundamental frequency is used to detect the transitions and it is computed using Praat by default. To use the RAPT algorithm change the "pitch method" variable in the function articulation_continuous.

The formant frequencies are computed using Praat.

When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file.

Script is called as follows

python articulation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]


examples:


python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true" "true"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "dynamic" "false" "true"

"""


from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pysptk
import scipy.stats as st
from articulation_functions import extractTrans, V_UV
import uuid
from util import write_to_csv,write_dynamic_to_csv

#sys.path.append('../kaldi-io')
#from kaldi_io import write_mat, write_vec_flt

sys.path.append('../praat')
import praat_functions

def plot_art(data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff):
    plt.figure(1)
    plt.subplot(311)
    t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
    if len(t)>len(data_audio):
        t=t[:len(data_audio)]
    elif len(t)<len(data_audio):
        data_audio=data_audio[:len(t)]
    plt.plot(t, data_audio, 'k')
    plt.ylabel('Amplitude', fontsize=14)
    plt.xlim([0, t[-1]])
    plt.grid(True)

    plt.subplot(312)
    t0=np.linspace(0.0,t[-1],len(F0))
    plt.plot(t0, F0, color='r', linewidth=2.0, label='F0')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.ylim([0,np.max(F0)+10])
    plt.xlim([0, t0[-1]])
    plt.grid(True)
    plt.legend()

    plt.subplot(313)
    fsp=int(len(F1)/t[-1])
    t2=np.arange(0.0, t[-1], 1.0/fsp)
    if len(t2)>len(F1):
        t2=t2[:len(F1)]
    elif len(F1)>len(t2):
        F1=F1[:len(t2)]
        F2=F2[:len(t2)]


    plt.plot(t2, F1, color='k', linewidth=2.0, label='F1')
    plt.plot(t2, F2, color='g', linewidth=2.0, label='F2')

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.ylim([0,np.max(F2)+10])
    plt.xlim([0, t2[-1]])
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.title("Onset segments")
    for j in range(len(segmentsOn)):
        plt.subplot(int(np.sqrt(len(segmentsOn)))+1, len(segmentsOn)/int(np.sqrt(len(segmentsOn))), j+1)
        t=np.arange(0, float(len(segmentsOn[j]))/fs, 1.0/fs)
        plt.plot(t, segmentsOn[j], linewidth=2.0)

    plt.show()

    plt.figure(3)
    plt.title("Offset segments")
    for j in range(len(segmentsOff)):
        plt.subplot(int(np.sqrt(len(segmentsOff)))+1, len(segmentsOff)/int(np.sqrt(len(segmentsOff))), j+1)
        t=np.arange(0, float(len(segmentsOff[j]))/fs, 1.0/fs)
        plt.plot(t, segmentsOff[j], linewidth=2.0)

    plt.show()


def articulation_continuous(audio_filename, flag_plots,sizeframe=0.04,step=0.02,nB=22,nMFCC=12,minf0=60,maxf0=350, voice_bias=-0.5,len_thr_miliseconds=270.0, pitch_method='praat'):

    fs, data_audio=read(audio_filename)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=sizeframe*float(fs)
    size_stepS=step*float(fs)
    overlap=size_stepS/size_frameS

    if pitch_method == 'praat':
        name_audio=audio_filename.split('/')
        temp_uuid='artic'+name_audio[-1][0:-4]
        temp_filename_vuv='../tempfiles/tempVUV'+temp_uuid+'.txt'
        temp_filename_f0='../tempfiles/tempF0'+temp_uuid+'.txt'
        praat_functions.praat_vuv(audio_filename, temp_filename_f0, temp_filename_vuv, time_stepF0=step, minf0=minf0, maxf0=maxf0)
        F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),step)
        segmentsFull,segmentsOn,segmentsOff,segments_onset_timelines=praat_functions.read_textgrid_trans(temp_filename_vuv,data_audio,fs,sizeframe)
        print(len(segments_onset_timelines),len(segmentsOn),len(segmentsOff))
        os.remove(temp_filename_vuv)
        os.remove(temp_filename_f0)
    elif pitch_method == 'rapt':
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=minf0, max=maxf0, voice_bias=voice_bias, otype='f0')
        segments= read_Textgrid(path_base+'vuv.txt', file_audio, win_trans)
        segmentsOn=V_UV(F0, data_audio, fs, 'onset')
        segmentsOff=V_UV(F0, data_audio, fs, 'offset')


    BBEon, MFCCon,time_lines=extractTrans(segmentsOn, fs, size_frameS, size_stepS,segments_onset_timelines, nB, nMFCC)
    BBEoff, MFCCoff,time_lines_unused=extractTrans(segmentsOff, fs, size_frameS, size_stepS,segments_onset_timelines, nB, nMFCC)

    DMFCCon=np.asarray([np.diff(MFCCon[:,nf], n=1) for nf in range(MFCCon.shape[1])]).T
    DDMFCCon=np.asarray([np.diff(MFCCon[:,nf], n=2) for nf in range(MFCCon.shape[1])]).T

    DMFCCoff=np.asarray([np.diff(MFCCoff[:,nf], n=1) for nf in range(MFCCoff.shape[1])]).T
    DDMFCCoff=np.asarray([np.diff(MFCCoff[:,nf], n=2) for nf in range(MFCCoff.shape[1])]).T

    # TODO: Make parameters configurable. (If worth it)
    name_audio=audio_filename.split('/')
    temp_uuid='artic'+name_audio[-1][0:-4]
    temp_filename='../tempfiles/tempFormants'+temp_uuid+'.txt'
    praat_functions.praat_formants(audio_filename, temp_filename,sizeframe,step)
    [F1, F2]=praat_functions.decodeFormants(temp_filename)
    os.remove(temp_filename)

    if len(F0)<len(F1):
        F0=np.hstack((F0, np.zeros(len(F1)-len(F0))))
    else:
        F1=np.hstack((F1, np.zeros(len(F0)-len(F1))))
        F2=np.hstack((F2, np.zeros(len(F0)-len(F2))))

    pos0=np.where(F0==0)[0]
    dpos0=np.hstack(([1],np.diff(pos0)))
    f0u=np.split(pos0, np.where(dpos0>1)[0])

    thr_sil=int(len_thr_miliseconds/step)

    sil_seg=[]
    for l in range(len(f0u)):
        if len(f0u[l])>=thr_sil:
            F1[f0u[l]]=0
            F2[f0u[l]]=0
        sil_seg.append(f0u)

    sil_seg=np.hstack(sil_seg)

    F1nz=F1[F1!=0]
    F2nz=F2[F2!=0]
    DF1=np.diff(F1, n=1)
    DF2=np.diff(F2, n=1)
    DDF1=np.diff(F1, n=2)
    DDF2=np.diff(F2, n=2)

    if flag_plots:
        plot_art(data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff)


    return BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1nz, DF1, DDF1, F2nz, DF2, DDF2,time_lines


if __name__=="__main__":
    prompt=' <file_or_folder_audio> <file_features.txt> '
    prompt+='[dynamic_or_static (default static)] '
    prompt+='[plots (true or false) (default false)] '
    prompt+='[kaldi output (true or false) (default false)] '
    if len(sys.argv)==6:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        if sys.argv[4]=="false" or sys.argv[4]=="False":
            flag_plots=False
        elif sys.argv[4]=="true" or sys.argv[4]=="True":
            flag_plots=True
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        if sys.argv[5]=="true" or sys.argv[5]=="True":
            flag_kaldi=True
        elif sys.argv[5]=="false" or sys.argv[5]=="False":
            flag_kaldi=False
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
    elif len(sys.argv)==5:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        if sys.argv[4]=="false" or sys.argv[4]=="False":
            flag_plots=False
        elif sys.argv[4]=="true" or sys.argv[4]=="True":
            flag_plots=True
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        flag_kaldi=False
    elif len(sys.argv)==4:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        flag_plots=False
        flag_kaldi=False
    elif len(sys.argv)==3:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        flag_static="static"
        flag_plots=False
        flag_kaldi=False
    elif len(sys.argv)<3:
        print('python '+sys.argv[0]+prompt)
        sys.exit()


    if audio.find('.wav')!=-1:
        nfiles=1
        hf=['']
    else:
        hf=os.listdir(audio)
        hf.sort()
        nfiles=len(hf)

    if flag_kaldi:
        Features={} # Kaldi Output requires a dictionary
        FeaturesOnset={}
        FeaturesOffset={}
    else:
        FeaturesOnset=[]
        IDon=[]
        FeaturesOffset=[]
        IDoff=[]
        Features=[]
        
    audio_names=[]
    audio_names_dynamic=[]
    time_lines_pool=[]

    for k in range(nfiles):
        audio_file=audio+hf[k]
        if audio_file.endswith(".wav"):
            filename_, file_extension_ = os.path.splitext(os.path.basename(audio_file))
            audio_names.append(filename_)
            
        print("Processing audio "+str(k+1)+ " from " + str(nfiles)+ " " +audio_file)

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2, time_lines=articulation_continuous(audio_file, flag_plots)

        time_lines_pool=time_lines_pool+time_lines[2:]
        
        if flag_static=="static":

            BBEon_=[]
            for i in range(1,23):
                string='BBEon'+str(i)+"_mean"
                BBEon_.append(string)
                string='BBEon'+str(i)+"_std"
                BBEon_.append(string)
                string='BBEon'+str(i)+"_sk"
                BBEon_.append(string)
                string='BBEon'+str(i)+"_ku"
                BBEon_.append(string)
                
            MFCCon_=[]
            for i in range(1,13):
                string='MFCCon'+str(i)+"_mean"
                MFCCon_.append(string)
                string='MFCCon'+str(i)+"_std"
                MFCCon_.append(string)
                string='MFCCon'+str(i)+"_sk"
                MFCCon_.append(string)
                string='MFCCon'+str(i)+"_ku"
                MFCCon_.append(string)  
                      
            DMFCCon_=[]
            for i in range(1,13):
                string='DMFCCon'+str(i)+"_mean"
                DMFCCon_.append(string)
                string='DMFCCon'+str(i)+"_std"
                DMFCCon_.append(string)
                string='DMFCCon'+str(i)+"_sk"
                DMFCCon_.append(string)
                string='DMFCCon'+str(i)+"_ku"
                DMFCCon_.append(string)
                
            DDMFCCon_=[]
            for i in range(1,13):
                string='DDMFCCon'+str(i)+"_mean"
                DDMFCCon_.append(string)
                string='DDMFCCon'+str(i)+"_std"
                DDMFCCon_.append(string)
                string='DDMFCCon'+str(i)+"_sk"
                DDMFCCon_.append(string)
                string='DDMFCCon'+str(i)+"_ku"
                DDMFCCon_.append(string)
           
            BBEoff_=[]
            for i in range(1,23):
                string='BBEoff'+str(i)+"_mean"
                BBEoff_.append(string)
                string='BBEoff'+str(i)+"_std"
                BBEoff_.append(string)
                string='BBEoff'+str(i)+"_sk"
                BBEoff_.append(string)
                string='BBEoff'+str(i)+"_ku"
                BBEoff_.append(string) 
                
            MFCCoff_=[]
            for i in range(1,13):
                string='MFCCoff'+str(i)+"_mean"
                MFCCoff_.append(string)
                string='MFCCoff'+str(i)+"_std"
                MFCCoff_.append(string)
                string='MFCCoff'+str(i)+"_sk"
                MFCCoff_.append(string)
                string='MFCCoff'+str(i)+"_ku"
                MFCCoff_.append(string)

            DMFCCoff_=[]
            for i in range(1,13):
                string='DMFCCoff'+str(i)+"_mean"
                DMFCCoff_.append(string)
                string='DMFCCoff'+str(i)+"_std"
                DMFCCoff_.append(string)
                string='DMFCCoff'+str(i)+"_sk"
                DMFCCoff_.append(string)
                string='DMFCCoff'+str(i)+"_ku"
                DMFCCoff_.append(string) 
                
            DDMFCCoff_=[]
            for i in range(1,13):
                string='DDMFCCoff'+str(i)+"_mean"
                DDMFCCoff_.append(string)
                string='DDMFCCoff'+str(i)+"_std"
                DDMFCCoff_.append(string)
                string='DDMFCCoff'+str(i)+"_sk"
                DDMFCCoff_.append(string)
                string='DDMFCCoff'+str(i)+"_ku"
                DDMFCCoff_.append(string)
                
            feature_title_name=BBEon_+ MFCCon_+ DMFCCon_+ DDMFCCon_+ BBEoff_+ MFCCoff_+ DMFCCoff_+ DDMFCCoff_+['F1_mean', 'F1_std','F1_sk','F1_ku', 
            'DF1_mean', 'DF1_std','DF1_sk','DF1_ku',
            'DDF1_mean', 'DDF1_std','DDF1_sk','DDF1_ku',
            'F2_mean', 'F2_std','F2_sk','F2_ku',
            'DF2_mean', 'DF2_std','DF2_sk','DF2_ku',
            'DDF2_mean', 'DDF2_std','DDF2_sk','DDF2_ku',]
            Feat=[BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2]

            Nfr=[Feat[n].shape[0] for n in range(len(Feat))]

            avgfeat=[]
            stdfeat=[]
            sk=[]
            ku=[]
            for n in range(len(Feat)):
                Nfr=len(Feat[n].shape)
                
                if Feat[n].shape[0]>1:
                    avgfeat.append(Feat[n].mean(0))
                    stdfeat.append(Feat[n].std(0))
                    sk.append(st.skew(Feat[n]))
                    ku.append(st.kurtosis(Feat[n], fisher=False))
                elif Feat[n].shape[0]==1:
                    avgfeat.append(Feat[n][0,:])
                    stdfeat.append(np.zeros(Feat[n].shape[1]))
                    sk.append(np.zeros(Feat[n].shape[1]))
                    ku.append(np.zeros(Feat[n].shape[1]))
                else:
                    avgfeat.append(np.zeros(Feat[n].shape[1]))
                    stdfeat.append(np.zeros(Feat[n].shape[1]))
                    sk.append(np.zeros(Feat[n].shape[1]))
                    ku.append(np.zeros(Feat[n].shape[1]))
                print(avgfeat)

                #print(len(avgfeat[-1]))

            Features_mean=np.hstack(avgfeat)
            Features_std=np.hstack(stdfeat)
            Features_sk=np.hstack(sk)
            Features_ku=np.hstack(ku)
            feat_vec=np.hstack((Features_mean, Features_std, Features_sk, Features_ku))
            if flag_kaldi:
                key=hf[k].replace('.wav', '')
                Features[key]=feat_vec
            else:
                Features.append(feat_vec)


        if flag_static=="dynamic":
                     
            BBEon_name=[]
            for i in range(1,23):
                string='BBEon'+str(i)
                BBEon_name.append(string)

            MFCCon_name=[]
            for i in range(1,13):
                string='MFCCon'+str(i)
                MFCCon_name.append(string)

            DMFCCon_name=[]
            for i in range(1,13):
                string='DMFCCon'+str(i)
                DMFCCon_name.append(string)
                
            DDMFCCon_name=[]
            for i in range(1,13):
                string='DDMFCCon'+str(i)
                DDMFCCon_name.append(string)
           
                    
            feature_title_name=['time_line']+BBEon_name+ MFCCon_name+ DMFCCon_name+ DDMFCCon_name
            
            feat_onset=np.hstack((BBEon[2:,:], MFCCon[2:,:], DMFCCon[1:,:], DDMFCCon))

            
            replicate_file_name= [filename_] * feat_onset.shape[0]
            audio_names_dynamic=audio_names_dynamic+replicate_file_name
            
            if flag_kaldi:
                if feat_onset.shape[0] > 0:
                    key=hf[k].replace('.wav', '')
                    FeaturesOnset[key]=feat_onset
            else:
                IDson=np.ones(feat_onset.shape[0])*(k+1)
                FeaturesOnset.append(feat_onset)
                IDon.append(IDson)
            feat_offset=np.hstack((BBEoff[2:,:], MFCCoff[2:,:], DMFCCoff[1:,:], DDMFCCoff))
            if flag_kaldi:
                if feat_offset.shape[0] > 0:
                    key=hf[k].replace('.wav', '')
                    FeaturesOffset[key]=feat_offset
            else:
                IDsoff=np.ones(feat_offset.shape[0])*(k+1)
                FeaturesOffset.append(feat_offset)
                IDoff.append(IDsoff)

    if flag_static=="static":
        if flag_kaldi:
            temp_file='temp_static_art'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(Features):
                    write_vec_flt(f, Features[key], key=key)
            ark_file=file_features.replace('.txt','')+'.ark'
            scp_file=file_features.replace('.txt','')+'.scp'
            os.system("copy-vector ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            #Features=np.asarray(Features)
            #print(file_features, Features.shape)
            #np.savetxt(file_features, Features)
            write_to_csv(audio_names,Features,file_features,feature_name=feature_title_name)

    if flag_static=="dynamic":
        if flag_kaldi:
            temp_file='temp_dynamic_art'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(FeaturesOnset):
                    write_mat(f, FeaturesOnset[key], key=key)
            ark_file=file_features.replace('.txt','')+'_onset.ark'
            scp_file=file_features.replace('.txt','')+'_onset.scp'
            os.system("copy-matrix ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            # os.remove(temp_file)
            temp_file='temp_dynamic_art2'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(FeaturesOffset):
                    write_mat(f, FeaturesOffset[key], key=key)
            ark_file=file_features.replace('.txt','')+'_offset.ark'
            scp_file=file_features.replace('.txt','')+'_offset.scp'
            os.system("copy-matrix ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            FeaturesOnset=np.vstack(FeaturesOnset)
            Features=[]
            for row in range(len(time_lines_pool)):
                Features.append([time_lines_pool[row]]+list(FeaturesOnset[row,:]))
            write_dynamic_to_csv(audio_names_dynamic,Features,file_features,feature_name=feature_title_name)
