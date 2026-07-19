# -*- coding: utf-8 -*-
"""Build public-release annotation files: Sample A (79-DWA fidelity) with notes
translated zh->en (all 102 notes), Sample B (150-action types) combined."""
import pandas as pd

T = {  # (annotator, item) -> English translation of the note
(1,3):"Steps 7-9 (delivering instruction and assessment) exceed the scope of the 'planning' activity itself",
(1,7):"Uses scaffolding as a concrete instance; coverage complete",
(1,13):"Steps 6-7 implement the optimisation and re-test, beyond 'assessment' scope",
(1,15):"Most steps are the firefighter personally extinguishing the fire rather than 'directing'; directing elements (task allocation, coordination) appear only in step 2",
(1,16):"Step 8 monitoring implementation effects slightly exceeds 'making recommendations', but is a reasonable extension",
(1,17):"Decomposed from an automation/vision-system perspective, but every step belongs to the activity",
(1,19):"Good overall; a few steps overly fragmented (approaching the truck, opening the door)",
(1,24):"Step 2 'diagnosing the condition' is an upstream activity preceding prescription; strictly beyond 'prescribing' itself",
(1,25):"Steps read as parallel knowledge-maintenance channels rather than a strict sequence, but all relevant",
(1,27):"Steps 7 and 9 (drafting the modification plan, follow-up) slightly exceed 'assessing the environment'",
(1,29):"Steps 6-8 (delivering activities and debriefing) exceed 'planning' scope",
(1,32):"Steps 5-6 (demonstrating and supervising exercises) lean toward delivering treatment, slightly beyond 'developing the programme'",
(1,33):"Step 8 monitoring production slightly exceeds 'planning' scope",
(1,36):"Steps 1 and 10 (arriving at/leaving court) slightly trivial; the rest appropriate",
(1,42):"Steps 8-10 (thanking, asking, farewell) fragmented and partly repetitive",
(1,50):"Steps 7-8 (reporting to the client and recording the decision) slightly exceed 'identifying opportunities'",
(1,56):"Uses a gear assembly as a concrete instance; steps reasonable",
(1,59):"Includes prototyping and testing, a reasonable part of the design-iteration loop",
(1,60):"Steps are mostly the detective personally investigating rather than 'directing', but content is relevant",
(1,65):"Steps 6-7 (helping obtain resources, following up) exceed 'assessing needs'",
(1,66):"Steps 8-9 (same-day monitoring and debrief) slightly exceed 'planning' scope",
(1,78):"Uses ride-hailing as a concrete instance; coverage complete",
(1,79):"Uses wheel-throwing as a concrete instance; coverage complete",
(2,1):"Step 4 summarising overlaps slightly with step 5 drafting",
(2,2):"'Navigate to the scene' is a padding step; the generic DWA is over-specified into a traffic-accident scene",
(2,3):"Steps 7-9 (teaching/assessing/reflecting) exceed 'planning'; implementation phase",
(2,4):"'Proceed to the equipment location' as a separate step is over-fragmented",
(2,7):"Title is generic 'temporary equipment or structures'; entire sequence specified into a scaffolding scenario",
(2,10):"'Navigate to the first machine' plus loop 'repeat steps 2-6' somewhat padded",
(2,11):"Too many mobility steps (navigation, house-to-house loop, return to start); over-fragmented",
(2,12):"'Log into the system' overly trivial",
(2,13):"Steps 6-7 implement optimisation and re-run, beyond 'assessment'; login/opening dashboard over-fragmented",
(2,15):"Title is 'direct'; most steps are the firefighter personally executing (donning PPE, connecting hoses, extinguishing); directing elements thin; scope drift",
(2,16):"Step 8 post-implementation monitoring slightly exceeds 'making recommendations'",
(2,17):"Fabricated 'vision-system automated workstation' setting; 'signal completion and await next workpiece' is a padding step",
(2,19):"'Walk to the truck avoiding obstacles', 'open the door', 'close the door' are classic fabricated ceremonial steps; over-fragmented",
(2,20):"'Guided relaxation/mindfulness practice' is fabricated specification of course content",
(2,21):"Contains loop 'repeat steps 3-7'",
(2,22):"'Log into the system' over-fragmented",
(2,23):"Step 7 sharing with colleagues slightly exceeds 'updating own knowledge'",
(2,24):"Step 2 'diagnosing the condition' is an upstream standalone activity, beyond 'prescribing' itself",
(2,26):"Steps 8-11 (writing materials, training staff, publishing, monitoring) exceed 'developing rules' into implementation",
(2,27):"Steps 7-9 jointly drafting the modification plan and following up slightly exceed 'assessment'",
(2,28):"Fabricated HVAC/tenant scenario; 'proceed to location' over-fragmented; 'directing' mixed with personally diagnosing and inspecting",
(2,29):"Steps 6-8 (briefing participants, leading activities, debriefing) are implementation, not 'planning'",
(2,30):"'Retrieve the tape measure from storage' over-fragmented, plus loop steps",
(2,31):"'Print it out', 'place on conveyor' scene details fabricated; steps 6 and 9 duplicate verification",
(2,32):"Fabricated post-knee-surgery rehabilitation scenario; steps 5-6 demonstrating and supervising exceed 'developing the programme'",
(2,33):"Steps 7-8 communication training/production monitoring slightly exceed 'planning'",
(2,34):"13 steps over-fragmented; 'ensure connections are secure' phrasing repeated in several places",
(2,36):"'Arrive at the courthouse and walk into the courtroom', 'politely exit the courtroom' are classic fabricated ceremonial steps",
(2,37):"Steps 1-3 (reviewing, drafting, approving policy) belong to policy formulation, preceding and beyond 'implementation'",
(2,39):"Generic 'heavy equipment' specified into an excavator; 'enter cab, fasten seatbelt, adjust seat' over-fragmented",
(2,41):"Specified into a programming/coding learning scenario",
(2,42):"Warm greeting, thanks, farewell and 'ensure they feel respected': many ceremonial fabricated steps; over-fragmented",
(2,43):"Fabricated 'library' operations scenario",
(2,44):"'Walk to the production line', 'open the entry system' over-fragmented; start/end time recording split into two steps; boundary between observation counting and 'recording information' blurred",
(2,46):"'Deliver to the archives department for custody' details fabricated; 12 steps long, some mergeable",
(2,48):"Steps 5-6 checking inventory/entering production system exceed 'reading the work order to determine specifications'; login/navigation over-fragmented",
(2,49):"Contains loop 'repeat steps 4-7'",
(2,50):"Steps 7-8 presenting to the client and recording the decision slightly exceed 'identifying opportunities/strategies'",
(2,52):"Step 7 periodic re-testing and step 8 multi-angle re-verification somewhat duplicate",
(2,53):"'Ensure suppliers are paid on time' exceeds logistics-coordination scope",
(2,56):"Fabricated specific gear-and-shaft assembly details (first gear, second gear, lock ring, press tool), inconsistent with the generic title",
(2,57):"'Illuminate hard-to-see areas with a flashlight' as a separate step somewhat fragmented",
(2,60):"Title is 'directing' an investigation; steps are the detective personally performing all examination and interviews; 'the first witness' narrative fabrication",
(2,64):"Fabricated 'robotic assembly line' scenario; 'log into the database at the terminal' over-fragmented",
(2,65):"Steps 6-7 connecting resources and ongoing follow-up exceed 'assessing needs' into case management",
(2,66):"Steps 8-9 same-day process monitoring and end-of-day debrief exceed 'planning' itself",
(2,67):"'Walk to the storeroom' over-fragmented; step 6 duplicates the verification at the end of step 5",
(2,68):"'Escort the patient to the testing area', 'thank and schedule follow-up' somewhat ceremonial; contains loop steps",
(2,71):"'Proceed to the equipment location' as a separate step over-fragmented",
(2,72):"Step 4 asking questions duplicates step 5 'pause the recording to ask'",
(2,75):"Generic 'green plumbing/water-handling systems' specified into a rain-barrel scenario",
(2,78):"Specified into a ride-hailing app dispatch scenario",
(2,79):"Title covers 'clay or dough'; only the pottery-wheel scenario covered",
(3,3):"Steps 7-8 (delivering activities, assessing students) are execution/assessment, not 'planning'; beyond activity scope",
(3,7):"Scaffolding as concrete instance; reasonable",
(3,8):"Missing safety-isolation steps before connection (source cut-off, pressure relief)",
(3,9):"Step 5 'securing' with fasteners/welding slightly exceeds 'positioning' itself",
(3,12):"Steps 1-2 (login, retrieval) fragmented; granularity uneven versus subsequent analysis steps",
(3,13):"Steps 6-7 implement optimisation and re-test, beyond 'assessment' scope",
(3,15):"Most steps are personal firefighting execution (PPE, extinguishing, search and rescue) rather than 'Direct'",
(3,16):"Step 8 post-implementation monitoring belongs to a later phase, slightly beyond 'making recommendations'",
(3,19):"Steps 1-2 (approaching the truck, opening the door) over-fragmented; granularity clearly uneven versus core inspection steps",
(3,24):"Step 2 'diagnosis' is upstream of prescribing; strictly beyond 'prescription' itself",
(3,26):"Steps 8-9 and 11 (materials, staff training, monitoring) lean to implementation, beyond 'developing rules'",
(3,27):"Steps 6-9 (recommendations, modification plan, follow-up) are intervention, not 'assessment'",
(3,29):"Steps 6-8 (briefing, leading activities, debrief) are implementation, not 'planning'",
(3,31):"Steps 6 and 9 duplicate verification; step 9 comes after loading onto the conveyor, ordering questionable",
(3,32):"Steps 5-6 demonstrating and supervising lean toward treatment delivery, slightly beyond 'developing the programme'",
(3,33):"Steps 7-8 (training communication, production monitoring) are implementation, beyond 'planning'",
(3,36):"First and last steps (arriving at court, leaving) fragmented",
(3,37):"Steps 1-3 (revising the draft policy and approving) are policy formulation; the DWA focuses on 'implementation'",
(3,42):"Steps 9-10 (asking about other needs, farewell) courtesy steps fragmented",
(3,45):"Single steps like 'write the code' are still large phases; granularity too coarse (related to the activity's own scale)",
(3,46):"Step 12 post-implementation monitoring/evaluation is a later phase; step 3 collection census also upstream",
(3,48):"Steps 5-6 (inventory check, system entry) slightly exceed 'reading the work order to determine requirements'",
(3,59):"Includes prototype building and live testing; a reasonable part of design iteration",
(3,60):"Steps mostly personal investigation (interviews, evidence collection) rather than 'directing', though lead investigators typically do both",
(3,65):"Steps 4-7 (developing the plan, connecting resources, follow-up) are service provision, beyond 'assessing needs'",
(3,66):"Steps 8-9 (same-day execution monitoring and debrief) are the execution phase, beyond 'planning'",
}

rows=[]
for a in [1,2,3]:
    x=pd.read_excel(f"sampleA_拆解质量_标注者{a}_已标注.xlsx")
    for _,r in x.iterrows():
        zh=str(r["notes"]).strip() if pd.notna(r["notes"]) else ""
        en=T.get((a,int(r["item"])),"") if zh else ""
        rows.append(dict(annotator=a,item=int(r["item"]),DWA_ID=r["DWA_ID"],
          Q1_complete=r["Q1_complete (1-5)"],Q2_faithful=r["Q2_faithful_no_hallucination (1-5)"],
          Q3_granularity=r["Q3_atomic_granularity_ok (1-5)"],notes_zh=zh,notes_en=en))
df=pd.DataFrame(rows)
miss=df[(df.notes_zh!="")&(df.notes_en=="")]
print("untranslated:",len(miss))
if len(miss): print(miss[["annotator","item","notes_zh"]].to_string())
df.to_csv("sampleA_fidelity_annotations_bilingual.csv",index=False,encoding="utf-8-sig")
print(f"saved sampleA_fidelity_annotations_bilingual.csv ({len(df)} rows, {(df.notes_zh!='').sum()} with notes)")

rb=[]
for a in [1,2,3]:
    x=pd.read_excel(f"sampleB_动作类型_标注者{a}_已标注.xlsx")
    x["annotator"]=a; rb.append(x)
pd.concat(rb).to_csv("sampleB_intelligence_type_annotations.csv",index=False,encoding="utf-8-sig")
print("saved sampleB_intelligence_type_annotations.csv")
