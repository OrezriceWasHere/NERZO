import re
from typing import List, Tuple,Dict
import json
# ----------------------------------------------------------------------
# core
# ----------------------------------------------------------------------


# ---------------------------------------------------------------------
def _is_open(text: str, i: int, inside: bool) -> bool:
    """
    Decide whether the '##' at position *i* opens or closes.
    Logic:
    ▸ Outside any entity  → always OPEN
    ▸ Inside an entity:
        – if the char **before** the hashes is alphanumeric → this is a CLOSE
        – else if the char **after** the hashes is alphanumeric → this is an OPEN
        – otherwise                                          → this is a CLOSE
    """
    if not inside:
        return True

    prev = text[i - 1] if i else ''           # char before '##'
    nxt  = text[i + 2] if i + 2 < len(text) else ''  # char after '##'

    if prev.isalnum():        # we’re at the end of an entity token
        return False          # → CLOSE
    if nxt.isalnum():         # hashes sit just before an entity token
        return True           # → OPEN
    return False              # default: CLOSE



def extract_entities(text: str, longest_only: bool = False
                     ) -> List[Tuple[str, int, int]]:
    clean, clean_pos, i = [], 0, 0
    stack: List[int] = []
    entities: List[Tuple[str, int, int]] = []

    while i < len(text):
        if text.startswith("##", i):
            if _is_open(text, i, stack):
                stack.append(clean_pos)           # OPEN
            elif stack:                           # CLOSE
                start = stack.pop()
                entities.append(("".join(clean[start:clean_pos]),
                                   start, clean_pos))
            i += 2
            continue

        clean.append(text[i])                     # copy char
        clean_pos += 1
        i += 1

    if longest_only:                              # keep only top-level
        outer = []
        for t, s, e in sorted(entities,
                              key=lambda x: x[2] - x[1],
                              reverse=True):
            if not any(s >= os and e <= oe for _, os, oe in outer):
                outer.append((t, s, e))
        return outer
    return entities
# ────────────────────────────────────────────────────────────
# 1.  MARKED-TEXT PARSER  (identical to previous version)
# ────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────
# 2.  ALIGNMENT helpers
# ────────────────────────────────────────────────────────────
def _find_next(hay: str, needle: str, start: int) -> Tuple[int,int]:
    k = hay.find(needle, start)
    return (k, k+len(needle)) if k != -1 else (-1, -1)

def _remove_nested(spans: List[Dict]) -> List[Dict]:
    """
    Throw away a span that is fully contained in (start >=,
    end <=) any longer span already kept.
    """
    keep: List[Dict] = []
    # longest → shortest ensures outer spans are kept first
    for sp in sorted(spans, key=lambda d: d['end']-d['start'], reverse=True):
        if not any(sp['start'] >= k['start'] and sp['end'] <= k['end']
                   for k in keep):
            keep.append(sp)
    # return them in document order
    return sorted(keep, key=lambda d: d['start'])

# ────────────────────────────────────────────────────────────
# 3.  MAIN: align marked → original
# ────────────────────────────────────────────────────────────
def align_to_original(marked: str, original: str,
                      longest_only: bool = False,
                      all_occurrences: bool = False
                      ) -> List[Dict]:
    """
    Locate every entity extracted from *marked* inside *original*.
    Shorter occurrences that fall wholly inside a longer span
    are automatically filtered out.
    """
    ents  = extract_entities(marked, longest_only)
    out   : List[Dict] = []
    cur   = 0

    for text, _, _ in ents:
        if all_occurrences:
            for m in re.finditer(re.escape(text), original):
                out.append({"text": text, "start": m.start(), "end": m.end()})
        else:
            b,e = _find_next(original, text, cur)
            if b != -1:
                out.append({"text": text, "start": b, "end": e})
                cur = e

    return _remove_nested(out)

if __name__ == "__main__":
    # example = ("####Claremore## Lake## is a reservoir in ##Rogers County##, ##Oklahoma##. Constructed in 1929-1930 by damming ##Dog Creek## for the purpose of providing water to the city of ##Claremore##, ##Oklahoma## and houses recreational amenities such as boat ramps, fishing docks, and picnic areas. In 2011, the lake added a 9-hole disc golf course.",
    #            "A further international service is provided by ##Venice Simplon Orient Express##. Although this is primarily a railtour operator, with special trains to various locations in the ##United Kingdom##, it also operates the scheduled ##Orient Express## service to destinations in ##Europe##. This involves two separate trains; the ##Belmond British Pullman## departs from ##London Victoria## and terminates at ##Folkestone West##, where passengers transfer by coach through the ##Channel Tunnel## to ##Calais##; at ##Gare de ##Calais####, they then join the ##Orient Express## which then calls at various destinations including ##Paris##, ##Vienna##, ##Innsbruck##, ##Venice## and ##Rome##.",
    #            "##Siem Reap## is a cluster of small villages along the ####Siem Reap## River##. These villages were originally developed around Buddhist pagodas (wats) which are almost evenly spaced along the river from ##Wat Preah En Kau Sei## in the north to ##Wat Phnom Krom## in the south, where the ####Siem Reap## River## meets the great ##Tonlé Sap Lake##.",
    #            "The production of wine declined significantly after the ##Muslim conquest of ##Egypt#### in the 7th century. Attitudes towards alcohol varied greatly under Islamic rule, Muslim rulers generally showed some level of tolerance towards alcohol production controlled by religious minorities. ##Jewish## manuscripts from the ##Cairo Geniza## recount the involvement of ##Egyptian## ##Jewish##s in the production and sale of wine in medieval ##Egypt##. The consumption of wine was not necessarily limited to religious minorities however. ##Western## travelers and pilgrims passing through ##Cairo## on their journeys reported that ##Muslim## locals imbibed on wine and a local barley beer, known as ##booza## (, not to be confused with the ##Levantine## ice cream of the same name), even during the most draconian periods of Islamic rule. The most popular wine was known as ##nebit shamsi## (), made from imported raisins and honey and left to ferment in the sun (hence the name, which roughly translates into sun wine).",
    #            "1992: ####South Australia#### 19.19.(133) d ##Victoria## 18.12.(120). ##Wayne Carey## (SA) described this game as the reason he believed he could succeed in the ##AFL##. In a high scoring game, ##Stephen Kernahan## (SA) kicked six goals, ##Paul Salmon## (Vic) kicked five and ##Paul Roos## (Vic) kicked three. ##Wayne Carey## dominated at centre half forward and kicked two goals. ####South Australia#### won the game in the final moments.",
    #            """assistant\nIn response to the ##Itamar## attack, on 13 March, the ##Israeli## cabinet approved the construction of 500 housing units in ##Gush Etzion##, ##Ma'ale Adumim##, ##Ariel## and ##Modi'in Illit##, areas of the ##West Bank## that ##Israel## intends to keep under any permanent accord with the ##Palestinians##. The decision was taken in a late-night cabinet meeting in which both Prime Minister ##Benjamin Netanyahu## and Defense Minister ##Ehud Barak## took part, after several alternatives, such as starting a new settlement or widening the settlement of ##Itamar##, were rejected. The decision was criticized by the ##Palestinians## and the ##United States##. A spokesperson from the U.S. State Department told the ##Jewish Week## that the ##United States## is deeply concerned by continuing ####Israel##i## actions with respect to settlements in the ##West Bank## and that [c]ontinued ####Israel##i## settlements are illegitimate and run counter to efforts to resume direct negotiations.""")
    #
    #
    # for text in example:
    #     original = text.replace("#", "")
    #     print(original)
    #     print(original)
    #     for ent in align_to_original(marked = text, original=original, longest_only=True,all_occurrences=True):
    #         print(ent)
    #         print(f"----{original[ent['start']:ent['end']]}----")
    #     print("\n" + "=" * 80 + "\n")
    #     # print(f"--- {text}... ---")
    #     # for ent in extract_entities(text, longest_only=True):
    #     #     print(f"{ent}: {original[ent[1]:ent[2]]} (from {ent[1]} to {ent[2]})")
    #     # print("\n" + "=" * 80 + "\n")

    # ────────────────────────────────────────────────────────────
    json_path = '/Users/urikatz2/phd/projects/AI2/code/AI2_projects/playground/nertrieve_span_extraction.json'
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for record in data.values():
        generated_text = record["generated"]
        original_text = record["sentence"]
        print(original_text)
        print(generated_text)
        for ent in align_to_original(marked = generated_text, original=original_text, longest_only=True,all_occurrences=True):
            print(ent)
            print(f"----{original_text[ent['start']:ent['end']]}----")
        print("\n" + "=" * 80 + "\n")