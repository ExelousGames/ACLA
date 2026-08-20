# Kestrel — AI Built Around the Application

Approximate spoken length: 10 minutes, including pauses for the on-screen demonstration.

## 0:00–0:50 — Opening

[Open on Kestrel. Move from the Live Session screen to a completed baseline recording, then briefly show the AI overlay.]

Most AI products begin with a blank chat box. You describe what you are looking at, copy information into the conversation, and hope the model has enough context to help.

Kestrel takes a different approach. The AI is built around the application.

It can work with the driving data that Kestrel already has, understand the baseline recording selected in the application, access the analysis produced from that baseline, and present useful information through the overlay.

The goal is not to add a chatbot beside a telemetry application. The goal is to connect the AI to the application’s real workflow—from recording a lap, to analyzing it, to delivering the result where the driver can use it.

In this demonstration, I’ll show the three parts available today: access to the baseline recording, analysis of that baseline, and presentation through the overlay.

## 0:50–1:55 — The application provides the context

[Show the session list and select a baseline recording. Open its playback and analysis workspace.]

The starting point is the application, not the prompt.

Here, I have selected a recorded baseline. Kestrel already knows which session is open, so the AI conversation can be associated with that recording. The recording carries the session context and the telemetry captured during the lap.

This matters because I should not need to explain which file I mean every time I ask a question. I should not need to export the telemetry, summarize the session by hand, or paste a chart into a generic assistant.

The selected baseline is the shared context between the driver, the application, and the AI.

[Start playback and move the playhead through the lap.]

The application can also track the current playback position and the active analyzed segment. That gives the interface a common reference point: the lap being reviewed, the position within that lap, and the performance section currently on screen.

The AI is therefore attached to a concrete object in Kestrel—the selected recording—rather than operating as a disconnected conversation.

## 1:55–3:15 — Giving the AI access to the baseline recording

[Open the AI Assistant while the baseline remains selected.]

When I open the AI Assistant here, it opens in the context of this recorded session.

I can ask a simple question such as:

[Ask: “What baseline recording am I looking at?”]

The important part of the response is not the wording. It is that the assistant can retrieve the session context from the application. It can identify the selected recording and work from the data associated with it.

That connection creates a much more natural workflow. I can refer to “this baseline,” “this lap,” or “the selected recording,” because the application has already established what those phrases mean.

[Show the transcript beside the selected session.]

The conversation also stays attached to the recorded-session screen. Moving between parts of Kestrel can change the assistant’s working context, so the AI follows the application instead of treating every screen as the same generic chat.

At this stage, the AI is not inventing an assessment from a lap time alone. It has access to the baseline recording that the driver is actually reviewing. That recording becomes the source material for the next step: analysis.

## 3:15–4:45 — Analyzing the baseline

[Ask: “Analyze this baseline recording.”]

From the same conversation, I can ask Kestrel to analyze the selected baseline.

[Show the analysis loading state, then open the results.]

The application sends the recorded data through its analysis pipeline and returns a structured result. Instead of reducing the entire lap to one number, the result divides the performance into track segments.

Each segment can contain its location in the recording, its track section, and the performance labels identified by the analysis. When expert reference data is available, Kestrel can also associate the driver’s segment with the corresponding expert data.

[Move through several analysis segments.]

This structure is important for the AI. It does not receive only a screenshot or a paragraph of generated prose. It can access the analysis result as application data.

That means the assistant can reason about the same segments the user can see in the interface. The visual analysis and the conversation are two views of the same result.

If the baseline has already been analyzed, the AI can use that existing result. If analysis has not been run yet, it can request the analysis from the application. In both cases, the workflow remains centered on the selected baseline.

## 4:45–6:15 — Turning analysis into a conversation

[Ask: “Which analyzed segment should I review first, and why?”]

Once the result is available, the AI can help interpret it.

I can ask which segment deserves attention, what labels were detected, or what the analysis says about a particular section of the lap. Because the assistant has access to the baseline analysis, the answer can be grounded in the result Kestrel produced.

[Select one segment in the results.]

For example, instead of asking a broad question such as “How can I drive faster?”, I can ask:

[Ask: “Explain the issue in this segment.”]

Or, when comparison data is available:

[Ask: “Show me how this section compares with the expert reference.”]

The AI can use the selected recording, the analyzed segment, and the available reference data to keep the discussion focused on the evidence in the application.

This is the core idea behind building AI around Kestrel. The user does not have to translate everything on screen into a prompt. The application exposes meaningful context to the assistant, and the assistant helps the user navigate and understand that context.

The AI adds a conversational layer to the analysis; it does not replace the analysis or hide its source.

## 6:15–7:35 — Showing the result in the overlay

[Enable the overlay. Keep the baseline analysis visible in the main application.]

The third part of the workflow is presentation.

Useful analysis should not be trapped inside the main Kestrel window. The AI can publish its response to the overlay, allowing the information to remain visible above the simulator or another application.

[Show an AI response appearing in the overlay.]

The overlay can carry the assistant’s message and the identity of the current AI session. This preserves the connection between the conversation in Kestrel and the information shown outside the main window.

For visual evidence, Kestrel can also present a driver-versus-expert comparison through the overlay.

[Show the selected driver-versus-expert comparison in the overlay.]

This is more useful than copying a text summary onto the screen. The driver can see the actual comparison associated with the analyzed baseline segment.

The main application remains the place where the recording and full analysis are explored. The overlay is the delivery surface: a focused view of the information the AI has chosen to bring forward.

## 7:35–9:05 — The complete workflow

[Show the full sequence without cuts where practical.]

Here is the current workflow from beginning to end.

First, I capture or select a baseline recording in Kestrel.

[Select the baseline.]

Second, I open that recording. The application establishes the session context for the AI, including the selected baseline and its playback state.

[Open the AI Assistant.]

Third, I ask the AI to analyze the baseline—or to use the analysis that is already available.

[Ask: “Analyze this baseline and identify the first section I should review.”]

Kestrel runs the analysis and organizes the result into segments. The assistant can access that structured result and explain which part of the baseline is relevant.

[Open the recommended segment and its comparison.]

Finally, I ask to see the useful result in the overlay.

[Ask: “Show this comparison in the overlay.”]

The selected guidance or comparison moves from the recorded-session workspace to the overlay, while remaining tied to the same AI conversation and the same baseline analysis.

The workflow is simple:

Record the baseline. Analyze the baseline. Discuss the result. Show the useful evidence in the overlay.

What makes it significant is that the AI participates in each step through the application’s own context. There is no manual data handoff between the telemetry, the analysis, the conversation, and the presentation layer.

## 9:05–9:35 — What the product supports today

[Show a three-part graphic: Baseline Recording → Baseline Analysis → AI Overlay.]

The current AI integration is deliberately focused.

Today, the assistant can work with the baseline recording selected in Kestrel, access or request the analysis for that baseline, and display supported responses and comparisons through the overlay.

That is the foundation. It establishes a reliable path from application data to AI reasoning to an in-context visual result.

As more parts of Kestrel are connected, they can follow the same pattern: expose real application state, give the AI a clear action, and render the result in the interface where it is most useful.

## 9:35–10:00 — Closing

[Return to the main application with the baseline analysis open and the comparison visible in the overlay.]

Kestrel’s AI is being built around the application—not added as a separate destination.

It can see the baseline recording the user is working with, use the analysis attached to that baseline, and carry the relevant result into the overlay.

The result is an AI experience grounded in the driver’s real data and connected to the way the application is already used.

That is the foundation for an AI performance analyst that understands the work, shows its evidence, and stays present where the driver needs it.

Thanks for watching.
