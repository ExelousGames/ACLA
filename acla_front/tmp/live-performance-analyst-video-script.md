# Kestrel — Live Performance Analyst

Approximate spoken length: 8 minutes, including pauses for the on-screen demonstration.

## 0:00–0:45 — Opening

[Open on Kestrel. Move from an active practice session to the Live Analyst conversation, a completed practice-lap analysis, and then the AI overlay.]

Most AI products begin with a blank chat box. You describe what you are looking at, copy information into the conversation, and hope the model has enough context to help.

Kestrel takes a different approach. The AI is built around the application.

It can work with the driving data that Kestrel already has, review a recorded practice lap, and present useful information through the overlay.

The goal is not to add a chatbot beside a telemetry application. The goal is to connect the AI to the application’s real workflow—from recording a practice lap, to analyzing it, to delivering the result where the driver can use it.

In this demonstration, I’ll show how to enable the Live Performance Analyst agent mode, how that mode focuses the AI on a live practice session and its practice laps, and how the resulting analysis can be presented through the overlay.

## 0:45–1:25 — The practice session provides the context

[Open Live Session, begin a practice recording, and complete a practice lap.]

The starting point is the application, not the prompt.

Here, Kestrel knows that I am in a live session. The active track, car, and practice laps become working context for the AI.

I do not need to export telemetry or describe the lap by hand. I can simply ask about the practice lap I have just driven.

The live practice session is the shared context between the driver, the application, and the AI.

[Briefly show the completed lap becoming available for analysis.]

## 1:25–2:50 — Enabling the Live Performance Analyst agent mode

[Keep the live practice recording active or paused. Open the AI Assistant and tap the mic to start the assistant connection.]

Kestrel’s main assistant can answer one-off live-session questions. For an ongoing performance review, I can switch to the dedicated Live Performance Analyst agent mode.

To enable it, I open the AI Assistant from Live Session, start the assistant connection with the mic, and say—or type:

[Ask: “Enable the Live Performance Analyst.”]

The main assistant starts a separate agent session. The interface identifies it as Live Analyst and pauses the main conversation while this focused mode is active.

This separation has a purpose. It allows the AI to concentrate on the practice session rather than divide its attention across the assistant’s general responsibilities. The analyst can stay focused on reviewing a practice lap, identifying patterns or mistakes, and turning the evidence into the next improvement target.

[Show the Live Analyst identity and the “Main Paused” state.]

At startup, the analyst asks what I want to analyze. I can give it a specific instruction such as:

[Say: “Analyze this practice lap and tell me what I should work on next.”]

For a longer practice run, I can instead ask it to review a few laps so it can look for repeated behavior rather than a single isolated moment. When I am finished, I can say “Stop the Live Performance Analyst” or select End Agent to return to the main assistant.

## 2:50–3:35 — A brief overview of practice-lap analysis

[Ask: “Analyze this practice lap and tell me what I should work on next.”]

The Live Performance Analyst takes the completed practice lap and runs Kestrel’s performance analysis.

[Briefly show the loading state, then open the results.]

The result presents the lap as track sections with the driving behaviors identified in each one. When expert reference data is available, Kestrel can also show a driver-versus-expert comparison.

[Move quickly through two analysis sections.]

The important point is not the internal analysis process. It is that the analyst can turn a practice lap into specific evidence about where the driver can improve.

The driver can request one lap for a focused review or a few laps to look for repeated behavior across the practice session.

## 3:35–4:50 — Turning analysis into a conversation

[Ask: “Which analyzed segment should I review first, and why?”]

Once the result is available, the Live Performance Analyst can help interpret it.

I can ask which section deserves attention or what the analysis says about a particular part of the lap. The answer stays grounded in the result Kestrel produced.

[Select one segment in the results.]

For example, instead of asking a broad question such as “How can I drive faster?”, I can ask:

[Ask: “Explain the issue in this segment.”]

Or, when comparison data is available:

[Ask: “Show me how this section compares with the expert reference.”]

The AI can use the analyzed section and available reference data to keep the discussion focused on the evidence in the application.

This is the core idea behind building AI around Kestrel. The user does not have to translate everything on screen into a prompt. The application exposes meaningful practice-session context to the analyst, and the analyst helps the user navigate and understand that context.

The AI adds a conversational layer to the result without hiding its source.

## 4:50–5:55 — Showing the result in the overlay

[Enable the overlay. Keep the lap analysis visible in the main application.]

The third part of the workflow is presentation.

Useful analysis should not be trapped inside the main Kestrel window. The AI can publish its response to the overlay, allowing the information to remain visible above the simulator or another application.

[Show an AI response appearing in the overlay.]

The overlay can carry the analyst’s message and the identity of the current AI session. This preserves the connection between the conversation in Kestrel and the information shown outside the main window.

For visual evidence, Kestrel can also present a driver-versus-expert comparison through the overlay.

[Show the selected driver-versus-expert comparison in the overlay.]

This is more useful than copying a text summary onto the screen. The driver can see the actual comparison associated with the analyzed lap section.

The main application remains the place where the recording and full analysis are explored. The overlay is the delivery surface: a focused view of the information the AI has chosen to bring forward.

## 5:55–6:55 — The complete workflow

[Show the full sequence without cuts where practical.]

Here is the current workflow from beginning to end.

First, I open Live Session and start recording my practice session.

[Start the practice recording and complete a lap.]

Second, I open the AI Assistant, start its connection, and ask it to enable the Live Performance Analyst.

[Open the AI Assistant.]

[Ask: “Enable the Live Performance Analyst.”]

The main assistant pauses, and the separate Live Analyst session takes over with a narrow job: review performance in this practice session.

Third, I tell the analyst what I want it to review.

[Ask: “Analyze this practice lap and identify the first section I should review.”]

Kestrel analyzes the practice lap, and the analyst explains which part deserves attention.

[Open the recommended segment and its comparison.]

Finally, I ask to see the useful result in the overlay.

[Ask: “Show this comparison in the overlay.”]

The selected guidance or comparison moves from the live-session analysis workspace to the overlay, while remaining tied to the same AI conversation and the same lap analysis.

The workflow is simple:

Record. Enable the Live Performance Analyst. Review the practice lap. Show the useful evidence in the overlay.

What makes it significant is that the AI participates in each step through the application’s own context. There is no manual data handoff between the telemetry, the analysis, the conversation, and the presentation layer.

## 6:55–7:25 — What the product supports today

[Show a four-part graphic: Practice Recording → Live Performance Analyst → Lap Analysis → AI Overlay.]

The current AI integration is deliberately focused.

Today, the main assistant can start the Live Performance Analyst during an active or paused live recording. That dedicated agent mode can focus on practice-session performance, review practice laps, and display supported guidance and comparisons through the overlay.

That is the foundation. It establishes a reliable path from application data to AI reasoning to an in-context visual result.

As more parts of Kestrel are connected, they can follow the same pattern: expose real application state, give the AI a clear action, and render the result in the interface where it is most useful.

## 7:25–8:00 — Closing

[Return to the main application with the lap analysis open and the comparison visible in the overlay.]

Kestrel’s AI is being built around the application—not added as a separate destination.

Its Live Performance Analyst agent mode can stay focused on the practice session, analyze the laps the driver is working on, and carry the relevant result into the overlay.

The result is an AI experience grounded in the driver’s real data and connected to the way the application is already used.

That is the foundation for an AI performance analyst that understands the work, shows its evidence, and stays present where the driver needs it.

Thanks for watching.
