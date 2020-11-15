#!/home/despoB/arjun/anaconda2/bin/python

import os, glob, sys
import numpy as np
import pandas as pd
import mne
import ecogtools
import matplotlib.pyplot as plt

def load_block(f, ef, discard_beg=8.0, discard_end=1.0, t_pre=1.5, t_post=1.5, hg_filter=False, apply_hilbert=False, resample=False, ofn_suff=''):
	'''This is a single large function that encapsulates all of the preprocessing steps we want to apply to each block.

	It was built by concatenating the cells of the ipython notebook 'Omission + Expectation', and should be used in conjunction with qsub to paralellize preproc.

	Parameters:
	f 		| filename of FIF file containing data
	ef 		| filename of events file

	Returns:

	'''
	# output file name
	block_str = f.split('/')[-2]
	ofn = 'omission%s-%s'%(ofn_suff,block_str)

	# bad channel setup
	sub_dir = '/home/knight/ECOGprediction/OmissionDetection/Data/Albany62/'
	bad_file = sub_dir + 'B03/BadChannels.csv'
	bads = pd.read_csv(bad_file)["badCh"].tolist()
	print "Bad channels are: %s"%bads

	# load data
	raw = mne.io.Raw(f, add_eeg_ref=False, preload=True)
	info = mne.create_info(raw.ch_names, raw.info['sfreq'], ch_types='misc')
	print 'There are %d timepoints.'%raw.n_times

	#creating raw array object
	raw_array = mne.io.RawArray(raw._data, info)

	# the following line only works if all channels have the same prefix!
	chname_pre = raw.ch_names[0][:-1] # first channel likely "<prefix> 1" - this gives "EEG "
	bad_chnames = [chname_pre + str(chnum) for chnum in bads]
	raw_array.info['bads'] = bad_chnames
	#print 'Bad Channels: ', ', '.join(raw.info['bads'])
	print info #, info['ch_names']

	# discard the first and last 10s, it takes a while for things to settle down
	# keep the indices as start and stop

	n_secs_total = raw.n_times/info['sfreq']
	print n_secs_total
	start, stop = raw_array.time_as_index([discard_beg, n_secs_total-discard_end])
	data, times = raw_array[:, start:stop]
	print(data.shape)
	print(times.shape, times.shape[0]/info['sfreq'])
	print times

	means = np.mean(data, axis=1)
	means = means[:, np.newaxis]

	print means.shape

	print raw_array._data[0,0] - raw_array._data[0,1], np.mean(raw_array._data[0,:])
	raw_array._data = raw_array._data - means
	print raw_array._data[0,0] - raw_array._data[0,1], np.mean(raw_array._data[0,:])

	# need to specify which channels to do our operations to.
	#good_chs = sorted(list(set(info['ch_names']).difference(set(info['bads']))))
	good_chs = [ch for ch in info['ch_names'] if ch not in info['bads']]
	print 'There are %d good channels.'%len(good_chs)
	#print good_chs, len(good_chs)
	#print good_chs, good_chs2, good_chs == good_chs2
	# get the number from the name, make it into an int, subtract 1 to get that channel's index
	# good_ch_idxs = sorted([int(x[len(chname_pre): ])-1 for x in good_chs])

	print raw_array._data.shape
	raw_array=raw_array.pick_channels(good_chs)
	print raw_array._data.shape
	n_good_chs = raw_array._data.shape[0]

	# Bandpass
	bp_l = 1.0
	bp_h = 190.0
	raw_array.filter(l_freq=bp_l, h_freq=bp_h, picks=range(n_good_chs))

	# Notch filter to remove line noise (60 Hz) and its harmonics at 120, 180, 240
	raw_array.notch_filter(np.arange(60,241,60), picks=range(n_good_chs), notch_widths=1)

	data, times = raw_array[:, start:stop]
	ch_sds = np.std(data, axis=1)
	print data.shape, ch_sds.shape
	mean_ch_sd = np.mean(ch_sds)
	print mean_ch_sd
	bad_sd_ch_idxs = [idx for idx in range(ch_sds.shape[0]) if ch_sds[idx] > 2*mean_ch_sd]
	bad_sd_ch_names = [info['ch_names'][i] for i in bad_sd_ch_idxs]
	print "# Bad channels: ", len(bad_sd_ch_idxs), bad_sd_ch_names

	raw_array = raw_array.drop_channels(bad_sd_ch_names)
	print raw_array._data.shape
	print data.shape # if we had dropped channels here, this wouldn't have been updated!
	print len(info['ch_names'])
	n_good_chs = raw_array._data.shape[0]

	# Subtract the common average reference
	data, times = raw_array[:, start:stop]
	print data.shape, data.T.shape
	car_data = ecogtools.car(data.T).T
	# replace the relevant interval in raw_array with the CAR'd data
	foo = raw_array._data[:, start:stop]
	print foo.shape, car_data.shape
	raw_array._data[:, start:stop] = car_data

	# Isolate only the frequencies we want to examine (high-gamma)
	if hg_filter:
		raw_array.filter(70, 150, picks=range(n_good_chs))

	if apply_hilbert:
		# beep boop
		# We'll also add zeros to our data so that it's of a length 2**N.
		# In signal processing, everything goes faster if your data is length 2**N :-)
		#range = raw_array.info['sfreq'] * 10
		next_pow2 = int(np.ceil(np.log2(raw_array.n_times)))
		print raw_array, raw_array.info, np.sum(raw_array._data < 0)
		raw_array.apply_hilbert(picks=range(n_good_chs), envelope=True,n_fft=2**next_pow2)
		print raw_array, raw_array.info, np.sum(raw_array._data < 0)
		raw_array.filter(None, 20, picks=range(n_good_chs)) #lowpass to remove noise
		print raw_array, raw_array.info, np.sum(raw_array._data < 0)

	# Load the events
	evs = mne.read_events(ef)
	print evs.shape

	#evs has a 3-col shape. first is onset, second is trigger_id(=event type), third is all-0s
	assert(all(evs[:,2]==0))
	# we need the trigger_ids in the last column (swap 2 and 3)
	evs[:,2] = evs[:,1]
	evs[:,1] = 0
	print evs[1:10,:]
	assert(all(evs[:,1]==0) & all(evs[:,2] != 0))

	omission_idxs = np.where(evs[:,2]==14)
	print omission_idxs[0]
	n_omissions = len(omission_idxs[0])
	for oidx in omission_idxs[0]:
	    t_idx = oidx - 3
	    if evs[t_idx,2] == 12:
	        new_trigger = 22
	    elif evs[t_idx,2] == 13:
	        new_trigger = 23
	    else:
	        continue
	    print "Changing omission at {oidx} to {nt}... ({t_idx} was {tt})".format(oidx=oidx, nt=new_trigger, t_idx=t_idx, tt=evs[t_idx,2])
	    evs[oidx, 2] = new_trigger

	# how many 14s are left?
	n_extra_omissions = sum(evs[:,2]==14)

	# triggers
	trigger_id = evs[:,2]
	print np.unique(trigger_id)
	ev_id = dict(la=11, ba=12, ga=13, ta=15, expected_ba=22, expected_ga=23)
	if n_extra_omissions > 0:
	    ev_id['omission'] = 14 # If this is included in the events dictionary but there are no events with this code, error
	print ev_id

	# use the provided onsets and event types to slice the data into epochs (=trials)
	# this constructor has many options! detrending can be applied here, for example
	# things to consider: detrending, eeg ref
	epochs = mne.Epochs(raw_array, evs, ev_id, -1*t_pre, t_post, detrend=None, add_eeg_ref=False, baseline=None)
	print "<0: ", np.sum(epochs._data < 0)

	ch_types = dict([(n, 'eeg') for n in info['ch_names']]) # 
	# print ch_types

	# This is probably going to give you n_channels Warnings, but if they say the unit was changed from NA to V...
	# ... that's actually what we're trying to do.
	# There appears to be a bug in the FIF reader, it didn't auto-populate the info structure with sensible values
	epochs.set_channel_types(ch_types)
	print epochs, epochs.info

	# call the function that excludes bad intervals. first, determine if there are any by looking for the file...
	bad_times_fn = os.path.join(sub_dir, block_str, 'BadSegments.csv')
	print bad_times_fn

	if os.path.exists(bad_times_fn):
		print 'Bad segments file exists...'
		# calculate bad intervals from file
		#bad_intervals = [(10, 15), (25, 30)] # dummy for now
		bt = np.array(pd.read_csv(bad_times_fn))
		bad_intervals = [(i[0], i[1]) for i in bt]
		epochs, evs = exclude_intervals(epochs, evs, t_pre, t_post, raw_array, bad_intervals)
	# else:
	# 	ev_onsets = evs[:,0]/1000.0 # event onsets in ms -- trial boundaries will be (this - t_pre) and (this + t_post)
	# 	ev_idxs = raw_array.time_as_index(ev_onsets.tolist())
	print "<0: ", np.sum(epochs._data < 0)
	epochs.load_data()
	print "<0: ", np.sum(epochs._data < 0)
	if resample: # either False or the desired sampling rate
		# print 'Resampling to: ', resample
		# print epochs._data.shape
		# epochs.resample(resample, copy=False)
		# epochs.info['sfreq'] = resample
		# info['sfreq'] = resample
		epochs.decimate(12)

	print "<0: ", np.sum(epochs._data < 0)
	t_min = -1 * t_pre
	#event_idx3 = np.vstack([ev_idxs, np.zeros_like(ev_idxs), np.zeros_like(ev_idxs)]).T
	#event_idx3 = np.vstack(ev_idxs, evs[:,1], evs[:,2])
	np.savez(ofn, data=epochs._data, info=epochs.info, events=evs, t_min=t_min, event_id=ev_id) # epochs=epochs_resamp, evs=evs, ev_id=ev_id)

def getOverlap(a, b): # helper function for exclude_intervals
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def exclude_intervals(epochs, evs, t_pre, t_post, raw_array, bad_intervals):
	'''bad_intervals should be a list of tuples.'''
	# get the indices at which each trial begins and ends
	# calculate how much to subtract/add to the event indices, then make an array with the bounds
	print 'In exclude_intervals...'
	print "<0: ", np.sum(epochs._data < 0)
	ev_onsets = evs[:,0]/1000.0 # event onsets in ms -- trial boundaries will be (this - t_pre) and (this + t_post)
	pre_idx = t_pre * epochs.info['sfreq']
	post_idx = t_post * epochs.info['sfreq']
	print pre_idx, post_idx
	ev_idxs = raw_array.time_as_index(ev_onsets.tolist())
	bounds = np.vstack([ev_idxs-pre_idx, ev_idxs+post_idx]).T.astype(int)

	print bounds.shape, bounds.dtype

	print bounds.shape
	drops = np.zeros(bounds.shape[0], dtype=bool)

	for i, trial in enumerate(bounds):
	    ba = trial.tolist()
	    #print 'Evaluating trial {i} with index bounds: {ba}...'.format(i=i, ba=ba)
	    for interval in bad_intervals:
	        t_beg = interval[0]
	        t_end = interval[1]
	        i_beg, i_end = raw_array.time_as_index([t_beg, t_end])
	        ol = getOverlap([i_beg, i_end], ba)
	        #print '{i} | {intl} => {ib}, {ie}: {ep} overlap with {ba}'.format(i=i, intl=interval, ib=i_beg, ie=i_end, ep=ol, ba=ba)
	        if ol > 0:
	            drops[i] = True
	            print 'Trial %d should be dropped because it falls within %s'%(i, str(interval))

	#print drops
	drop_idxs = np.where(drops==True)[0]
	print drop_idxs, len(drop_idxs)

	epochs.drop_epochs(drops, reason='USER: Bad time interval')
	bounds = bounds[~drops, :]
	ev_idxs = ev_idxs[~drops]
	evs = evs[~drops, :]
	print epochs
	print "<0: ", np.sum(epochs._data < 0)

	return (epochs, evs)

def build_classifier(all_epochs, event_ids, chan_name, use_events, t_start=None, t_stop=None, n_cv=5):
	'''Build a logistic regression classifier using data from one channel only.

	all_epochs: epochs object including all epochs and event types. not an average!
	event_ids: dict mapping event labels to numeric codes (which will be used as targets for classification)
	chan_name: list containing name(s) of channel(s) to be used, e.g. ['EEG 1']
	use_events: the events we want to classify. should be a list, entries are a subset of event_ids.keys()
	t_start, t_stop: times, in seconds relative to the event at the center of the epoch, to use data in between for classification
	'''
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import cross_val_score

	if not isinstance(chan_name, list):
		raise TypeError('chan_name must be a list containing the name(s)')
	chan_idx = mne.pick_channels(all_epochs.ch_names, chan_name).tolist()
	print 'Channel index(es): ', chan_idx

	# lists to hold data and targets for each event type
	traindata = []
	traintargets = []

	# figure out the indices that delimit (in time) the data to use for classification
	if t_start is not None:
		start_idx = int(round((t_start-all_epochs.tmin)*all_epochs.info['sfreq']))
	else:
		start_idx = 0
	if t_stop is not None:
		stop_idx = int(round((t_stop-all_epochs.tmin)*all_epochs.info['sfreq']))
	else:
		stop_idx = len(all_epochs.times)
	use_interval = range(start_idx,stop_idx)
	print 'Classifying from indexes %d:%d'%(start_idx, stop_idx)

	# if there's >1 channel we want to use, reshape the n_trials x n_ch x n_tps to n_trials x (n_ch x n_tps)
	# this corresponds to n_trials observations of a larger number of features, namely 'Channel X at this timepoint', not just 'this timepoint'
	for ev_name, ev_code in event_ids.iteritems():
		if ev_name in use_events:
		    print ev_name, ev_code
		    #print all_epochs[ev_name]._data.shape
		    n_ep_ev = all_epochs[ev_name]._data.shape[0]
		    n_tps = len(use_interval)
		    data_ev = (all_epochs[ev_name]._data[:, chan_idx])[:,:,use_interval] # all trials, selected channel(s)... then get selected timepoints
		    print data_ev.shape
		    data_ev = np.reshape(data_ev, (n_ep_ev, len(chan_idx)*n_tps)) # should do the same as np.squeeze() if only one channel specified
		    targets_ev = np.ones(n_ep_ev)*ev_code
		    print data_ev.shape, targets_ev.shape #targets_ev
		    traindata.append(data_ev)
		    traintargets.append(targets_ev)

	td = np.vstack(traindata)
	tt = np.hstack(traintargets)
	print td.shape, tt.shape

	#print len(traindata[0]), traintargets[0], len(traintargets[0])

	model = LogisticRegression(C=1)
	scores = cross_val_score(model, td, tt, cv=n_cv)
	print scores

	return model, scores

def mean_plot(scores, ev_t=None):
	mean_scores = np.mean(scores,axis=1)
	f = plt.figure(1)
	ax = f.add_subplot(1, 1, 1)
	plt.bar(np.arange(len(mean_scores)),mean_scores)
	if ev_t is not None:
		ax.axvline(x=ev_t)

def sup_plot(evokeds, chans, tstart, tstop, ev_t):
    '''function to plot the traces of different events for a particular channel superimposed.
    
    evokeds: dict, keys correspond to event_ids and values are Evoked average objects, n_channels x n_tps
    chans: list of channel indices to plot (each one gets its own subplot) (max length: 16)
    '''

    n_ch = len(chans)
    ch_names = evokeds[evokeds.keys()[0]].info['ch_names']
    #n_rc = int(np.ceil(n_ch**.5))
    n_cols = 3
    n_rows = (n_ch/n_cols)+1
    print n_ch, n_rows, n_cols
    f = plt.figure(figsize=(15, 4*n_rows))
    for i in range(int(n_ch)):
        if i < n_ch:
            ax = plt.subplot(n_rows, n_cols, i+1)
            for evt in evokeds.keys():
                evdata = evokeds[evt].data
                plt.plot([tp for tp in range(evdata.shape[1])], evdata[chans[i],:], label=evt)
            ax.axvline(x=ev_t)
            plt.title('%s (index %d)'%(ch_names[chans[i]],chans[i]))
    	if i%n_cols == n_cols-1:
    		plt.legend(bbox_to_anchor=(1.4,1.05))

if __name__ == '__main__':
	f = sys.argv[1]
	ef = sys.argv[2]

	# seconds to ignore at beginning and end
	discard_beg = 8.0 # seconds
	discard_end = 1.0 # seconds

	# time before and after each event that we want to capture as an epoch (trial)
	t_pre = 1.5
	t_post = 1.5

	load_block(f, ef, discard_beg=discard_beg, discard_end=discard_end, t_pre=t_pre, t_post=t_post, hg_filter=False, apply_hilbert=False, resample=True, ofn_suff='_erp')
