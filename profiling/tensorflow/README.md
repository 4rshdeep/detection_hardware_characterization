## Profiling a tensorflow code
Tensorflow has built in support which makes it very easy to profile a frozen graph in tensorflow.

In this repo we are going to profile `ssd_mobilenet_v1.pb`, by following the below mentioned steps:-
1. Import timeline by writing this at the very top of your tensorflow inference script
    ```python
    from tensorflow.python.client import timeline
    ```
2. Add these lines just before calling sess.run()
    ```python
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    ```
3. Call sess.run(), with `options` and `run_metadata` parameters
    ```python
    sess.run(res, options=options, run_metadata=run_metadata)
    ```
4. After sess.run(), write this code, to store the timeline as a chrome trace
    ```python
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
      f.write(chrome_trace)
    ```
5. To view this data, open chrome browser and go to `chrome://tracing`.I
6. In the upper left corner, you will find Load button. Press it and load our JSON file.

## Notes
* This method stores the data of one run in a chrome trace

## Credits
* [Illarion Khlestov](https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d)
