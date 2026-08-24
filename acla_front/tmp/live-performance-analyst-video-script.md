# Kestrel — Live Performance Analyst

Approximate spoken length: 8 minutes, including pauses for the on-screen demonstration.

## Opening

[Start a practice lap and call for live performance analyst. ask the ai to watch for a lap]

[Talk to AI: Can you do analysis on me.]
[Talk to AI: Live performance analyst please.]
[Talk to AI: i will do 1 lap, and analysis that lap for me]

[The AI starts the goal feature. ]

This is Kestrel, the sim racing data analysis assistant. I just used its Live Performance Analyst agent. It a instrcuted agentic ai that works by communicating with sensors built inside Kestrel and well equipped racing knowledge database. It turns telemetry graphs into language you can understand. This is not another ai coaching program, this is the assitant that can turns youself into your own racing analyst.

In this demonstration, I’ll show the complete workflow—from recording a lap, to finding the section that needs attention, to carrying that insight back onto the track.


## Enabling the Live Performance Analyst agent mode

[Keep the live practice recording active or paused. Open the AI Assistant and tap the mic to start the assistant connection.]

Kestrel’s live session assistant can answer one-off live-session questions. For an ongoing performance review, I can switch to the dedicated Live Performance Analyst agent mode.

To enable it, i am able to ask

[Ask: “Can you do a lap analysis for me”]
[AI: response]
[Ask: "start the live analyst"]

The main assistant starts a separate agent session. The interface identifies it as Live Analyst and pauses the main conversation while this focused mode is active.

This separation has a purpose. It allows the AI to concentrate on the pratice session's lap analyst. The analyst can stay focused on identifying patterns or mistakes.

[At startup, the analyst asks what I want to analyze. I can give it a specific instruction such as

Say: i will do one lap, then you will do the analysis.”]

the ai understand the request, and provide the sequence of recording the baseline, and analyse the baseline. If you like it can also do a longer practice run. I can instead ask it to review a few laps.

[Ask: do multiple laps]

In fact, i can adjustify the baseline recording condition a lot more. the recording starting condition and end condition are modifiable. there are some presets in the UI, but AI will able to tune it more by controlling the internal setting. I can start the recording when i throttle, and end when the car is at 100km/h.This is not the focus today. I would like to make a separated video for it.

When I am finished, I can say “Stop the Live Performance Analyst” or select End Agent to return to the main assistant.

## A brief overview of practice-lap analysis

The AI goes to analysis the laps as I requested.  AI will compare driver's recorded lap and reference lap, and it is trained to recongize the difference between driver and expert and associate it with a label. 

[Briefly show the loading state, then open the results.]

The analysis result presents the lap as track sections with the driving behaviors identified in each one. user will be able to hover each segment and see a animated driver-versus-expert comparison.

[Move quickly through two analysis sections.]

There are more i want to show, like how to use preset to manage the list, check the over trend, or ask AI to clean up the list. but, I would like to make separated video on this.


## Showing the result in the overlay

[Enable the overlay. Keep the lap analysis visible in the main application.]

Useful analysis result can be show outside the main Kestrel window. By design, the overlay can actual display any info available in the main window.

[Show an AI response appearing in the overlay.]

I showed that I can check a mistake by hovering over a result. But, I think display the comparsion graphs into the overlay while driving could be a good idea. With a simple comment, you can ask Kestrel to present each mistakes while driving. if no specific page asked, Kestrel will default to present the current displayed analysis result.

[Ask AI: i would like to see the mistake with detailed mistakes marked while driving, Show the selected driver-versus-expert comparison in the overlay.]

I think this is more useful than copying a text summary onto the screen. The driver can see the actual comparison associated with the analyzed lap section.

## The complete workflow

[Show the full sequence without cuts where practical.]

Thats one of the workflow of the live performance analst. 

I asked AI Assistant to enable the Live Performance Analyst.

[Show: “Enable the Live Performance Analyst.”]

The main assistant pauses, and the separate Live Analyst session takes over with a narrow job: review performance in this practice session.

I then tell the analyst what I want it to review.

[Show: “Analyze this practice lap and identify the first section I should review.”]

Kestrel analyzes the practice lap, and display in the analysis result

[Open the recommended segment and its comparison.]

Finally, I ask to see the useful result in the overlay.

[Show: “Show me mistakes comparison in the overlay.”]

This is one of workflow i worked on. but i would like to bring more feature to this agent mode.

## Closing

Kestrel’s AI is being built around the application—not added as a separate destination.

Its Live Performance Analyst agent mode can stay focused on the practice session, analyze the laps the driver is working on, and carry the relevant result into the overlay.

That is the foundation for an AI performance analyst that understands the work, shows its evidence, and stays present where the driver needs it.


