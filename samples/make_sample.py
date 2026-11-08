"""Writes samples/the-last-heir/: a short ORIGINAL fictional show in the same screenplay format as the assignment
pack (timestamped CAMERA / ACTION / dialogue / SOUND FX blocks), so anyone can try the demo without the
confidential scripts. Run: python samples/make_sample.py
"""
import os

# Each event: (kind, who_or_None, tags, text, seconds). kind: C=camera, A=action, D=dialogue, S=sound fx
EPISODES = [
    ("A waitress at a charity gala is publicly humiliated by her stepsister and fired on the spot, until the quiet chauffeur steps in.",
     "First episode. Establishes Mara, her cruel stepsister Vivian, and Elias, the chauffeur nobody notices.",
     [("INT. GRAND BALLROOM - NIGHT", [
         ("C", None, [], "Wide shot of a glittering ballroom; Mara weaves between guests with a tray of champagne.", 3.0),
         ("A", None, [], "Vivian, in a red gown, deliberately steps into Mara's path; the tray crashes to the floor.", 2.5),
         ("S", None, [], "Shattering glass silences the room.", 2.0),
         ("D", "VIVIAN", ["Loud, Mocking"], "\"Oh no. The help can't even carry a tray.\"", 2.5),
         ("D", "MARA", ["Shaken, Quiet"], "\"You tripped me. Everyone saw it.\"", 2.5),
         ("D", "VIVIAN", ["Cold, Smiling"], "\"Everyone saw a waitress embarrass my father's gala.\"", 3.0),
         ("C", None, [], "Close-up on Mara's trembling hands, glass shards around her shoes.", 2.0),
         ("D", "EVENT MANAGER", ["Furious, Shouting"], "\"Mara! Get out. You're fired!\"", 2.5),
         ("D", "MARA", ["Desperate, Pleading"], "\"Please, I need this job. My mother's hospital bills...\"", 3.5),
         ("D", "VIVIAN", ["Cruel, Delighted"], "\"Then maybe your mother should have raised a daughter who knows her place.\"", 4.0),
         ("A", None, [], "Guests laugh; someone raises a phone to film Mara on her knees.", 2.5),
         ("C", None, [], "Slow push-in on the ballroom doors as a tall man in a chauffeur's coat walks in.", 3.0),
         ("D", "ELIAS", ["Low, Calm"], "\"Put the phone down.\"", 2.0),
         ("D", "VIVIAN", ["Scornful"], "\"Who let the driver in? Go wait by the car.\"", 2.5),
         ("A", None, [], "Elias kneels and helps Mara to her feet, ignoring Vivian completely.", 3.0),
         ("D", "ELIAS", ["Gentle"], "\"Are you hurt?\"", 1.5),
         ("D", "MARA", ["Whispering, Confused"], "\"Why are you helping me?\"", 2.0),
         ("D", "ELIAS", ["Quiet, Certain"], "\"Because nobody else in this room will.\"", 2.5),
         ("D", "VIVIAN", ["Shrill, Outraged"], "\"Security! Throw them both out!\"", 2.5),
         ("A", None, [], "Two guards move in. Elias steps in front of Mara and holds up a single black card.", 3.0),
         ("C", None, [], "Extreme close-up on the card: a gold crest. The guards freeze.", 2.0),
         ("D", "GUARD", ["Stunned, Whispering"], "\"Sir... we didn't know it was you.\"", 2.5),
         ("D", "VIVIAN", ["Shaken"], "\"Know what? He's a driver.\"", 2.0),
         ("S", None, [], "A low string sting holds on Vivian's frozen face.", 2.5),
     ])]),
    ("Vivian tries to have Mara arrested for stealing a diamond bracelet, and Elias is forced to choose between his secret and the woman he defended.",
     "Picks up after the gala; the guards' reaction to Elias's card is unexplained.",
     [("EXT. HOTEL ENTRANCE - NIGHT", [
         ("C", None, [], "Rain on the hotel steps. Mara hurries out, coat over her uniform.", 3.0),
         ("D", "VIVIAN", ["Shouting, Triumphant"], "\"Stop her! She stole my bracelet!\"", 2.5),
         ("A", None, [], "A police officer blocks Mara's way at the bottom of the steps.", 2.5),
         ("D", "MARA", ["Panicked"], "\"I never touched her bracelet!\"", 2.0),
         ("D", "OFFICER", ["Stern"], "\"Then you won't mind if I check your bag.\"", 2.5),
         ("A", None, [], "The officer opens Mara's bag and lifts out a diamond bracelet.", 3.0),
         ("S", None, [], "Gasps from the guests gathered under the awning.", 2.0),
         ("D", "MARA", ["Horrified, Weeping"], "\"Someone put that there. I swear it!\"", 3.0),
         ("D", "VIVIAN", ["Smug, Cold"], "\"Thieves always swear.\"", 2.0),
         ("D", "OFFICER", ["Firm"], "\"Turn around. You're under arrest.\"", 2.5),
         ("C", None, [], "Close-up on the handcuffs closing on Mara's wrists.", 2.0),
         ("D", "ELIAS", ["Low, Dangerous"], "\"Check the hotel cameras before you take her anywhere.\"", 3.0),
         ("D", "VIVIAN", ["Mocking"], "\"Listen to the chauffeur giving orders.\"", 2.5),
         ("D", "ELIAS", ["Calm, Cold"], "\"The cameras are mine. I own this hotel.\"", 3.0),
         ("A", None, [], "The officer looks from Elias to Vivian; Vivian's smile slips.", 2.5),
         ("D", "VIVIAN", ["Stammering"], "\"That's impossible. The Halden family owns this hotel.\"", 3.0),
         ("D", "ELIAS", ["Quiet"], "\"Yes. They do.\"", 2.0),
         ("D", "MARA", ["Whispering, Stunned"], "\"Elias... who are you?\"", 2.5),
         ("A", None, [], "A black car pulls up; an old man in a suit steps out and stares at Elias.", 3.0),
         ("D", "OLD MAN", ["Grave, Shaken"], "\"Five years, Elias. Your father is dying, and you come back as a driver?\"", 4.0),
         ("S", None, [], "Thunder cracks over the street as the screen cuts to black.", 2.5),
     ])]),
    ("At the Halden mansion, Mara learns that Elias is the missing heir, and Vivian arrives with a secret that could destroy them both.",
     "Follows the arrest scene; Elias has brought Mara to the Halden estate.",
     [("INT. HALDEN MANSION - STUDY - NIGHT", [
         ("C", None, [], "Firelight in a vast study. Mara stands in her wet uniform, staring at a family portrait.", 3.0),
         ("D", "MARA", ["Quiet, Hurt"], "\"That's you. In the portrait. You lied to me.\"", 3.0),
         ("D", "ELIAS", ["Tired, Honest"], "\"I left this family five years ago. I didn't want their name.\"", 3.5),
         ("D", "MARA", ["Bitter"], "\"So you drove rich people around for fun?\"", 2.5),
         ("D", "ELIAS", ["Low"], "\"I was looking for the person who ruined my mother.\"", 3.0),
         ("A", None, [], "Elias slides a photograph across the desk toward Mara.", 2.5),
         ("C", None, [], "Close-up on the photograph: a young Vivian beside a burning car.", 2.0),
         ("D", "MARA", ["Shocked, Breathless"], "\"That's Vivian.\"", 2.0),
         ("S", None, [], "The study doors bang open.", 1.5),
         ("D", "VIVIAN", ["Icy, Confident"], "\"Showing her old pictures, Elias? I brought a newer one.\"", 3.5),
         ("A", None, [], "Vivian holds up a document stamped with a hospital seal.", 2.5),
         ("D", "VIVIAN", ["Cruel, Savoring"], "\"Mara's mother isn't sick. She's in prison, for the fire that killed yours.\"", 4.5),
         ("C", None, [], "Close-up on Mara, the color draining from her face.", 2.0),
         ("D", "MARA", ["Trembling, Disbelieving"], "\"That's a lie.\"", 2.0),
         ("D", "VIVIAN", ["Smug"], "\"Ask him why he really followed you.\"", 2.5),
         ("A", None, [], "Mara turns slowly toward Elias, waiting for him to deny it.", 3.0),
         ("D", "ELIAS", ["Quiet, Pained"], "\"Mara... I can explain.\"", 2.5),
         ("D", "MARA", ["Broken, Furious"], "\"You knew. From the very first night, you knew who I was.\"", 3.5),
         ("S", None, [], "A deep string swell holds on Mara backing away toward the door.", 3.0),
     ])]),
]


def tc(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def write(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for n, (logline, notes, scenes) in enumerate(EPISODES, 1):
        lines = [f"EPISODE NUMBER: Episode {n}", f"EPISODE LOGLINE: {logline}", f"CONTINUITY NOTES: {notes}", "",
                 "--- SCRIPT START ---", ""]
        t = 0.0
        for i, (heading, events) in enumerate(scenes, 1):
            start = t
            body = []
            for kind, who, tags, text, dur in events:
                label = {"C": "CAMERA", "A": "ACTION", "S": "SOUND FX"}.get(kind) or (who + (f" ({', '.join(tags)})" if tags else ""))
                body += [f"**{label}:**", f"Start: ({tc(t)}) | End: ({tc(t + dur - 0.1)})", text, ""]
                t += dur
            lines += [f"SCENE {i:02d}", f"Scene heading: {heading}", f"Scene start: {tc(start)}", f"Scene end: {tc(t)}", "",
                      "Actions and dialogue:", ""] + body
        lines += ["--- SCRIPT END ---"]
        with open(os.path.join(out_dir, f"episode-{n:03d}-the-last-heir.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    write(os.path.join(os.path.dirname(__file__), "the-last-heir"))
    print("wrote", len(EPISODES), "episodes")
