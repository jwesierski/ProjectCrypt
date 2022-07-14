![OpenBCI Headset](https://static.wixstatic.com/media/85d932_b4959a41ef9f48a3ae36b4ae15a516c5~mv2.jpg/v1/fill/w_1500,h_842,al_c,q_90/85d932_b4959a41ef9f48a3ae36b4ae15a516c5~mv2.webp)

# Brain-Computer Interface workspace

Recently, my team and I purchased ourselves a hat. 

However, not just any ordinary hat. It won't shade you from the sun nor complete an outfit (though maybe one day). Instead, we acquired it because it does something much more important: it can read your mind. The problem is we don’t yet understand the language. So I asked myself (and a few others):


# Question

How can patterns in the brain activity corresponding to thoughts be translated into meaningful sentences?


# Answer

We came to the conclusion that with a machine learning algorithm and Brain-Computer Interface we can use the associated brain activity of spoken words to translate thoughts into coherent sentences in real-time. 

For a full breakdown of our approach to this philosophical problem (and who _we_ are) check out journal article: https://www.amindapplied.com/post/reading-the-mind-preface. 

Some previous remarks related to the files in this discord:

`training.py` - This is merely an example of training a model with this data. . 

`analysis.py` - You can use this to run through validation data to see confusion matricies for your models on out of sample data.


`testing_and_making_data.py` - This is just here if you happen to have your own OpenBCI headset and want to actually play with the model and/or build on the dataset. Or if you just want to help audit/improve my code. This file will load in whatever model you wish to use, you will specify the action you intend to think ahead of time for the `ACTION` var, then you run the script. The environment will pop up and collect all of your FFT data, storing them to a numpy file in the dir named whatever you said the `ACTION` thought was.

For full directions check sentdexes...

# Requirements
Numpy
TensorFlow 2.0. (you need 2.0 if you intend to load the models)
pylsl (if you intend to run on an actual headset)
OpenBCI GUI (using the networking tab https://docs.openbci.com/docs/06Software/01-OpenBCISoftware/GUIDocs)



# The data

Currently, the data available is 16-channel FFT 0-60Hz, sampled at a rate of about 25/second. Data is contained in directories labeled as `left`, `right`, or `none`. These directories contain numpy arrays of this FFT data collected where I was thinking of moving a square on the screen in this directions. 

Each file is targeted to be 10 seconds long, which, at 25 iter/sec gives us, the 250 (though you should not depend/assume all files will be exactly 250 long). Then you have the number of channels (16), and then 60 values, for up to 60Hz. For example, if you do: 

