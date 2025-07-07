import re
from typing import List, Tuple,Dict
import json
# ----------------------------------------------------------------------
# core
# ----------------------------------------------------------------------


# ---------------------------------------------------------------------
def _is_open(text: str, i: int, inside: bool) -> bool:
    """Return True if the ## at position i should OPEN a span."""
    if not inside:                      # stack is empty → first ## always opens
        return True
    nxt = i + 2
    return nxt < len(text) and text[nxt].isalnum()


def extract_entities(text: str, longest_only: bool = False
                     ) -> List[Tuple[str, int, int]]:
    """
    Parse a sentence where entities are delimited with ## … ##,
    handling patterns like ####Claremore## Lake##.

    Returns [(entity_text, start, end)], offsets refer to the
    text *after* all ## markers are stripped.
    """
    clean       : List[str] = []
    clean_pos   : int       = 0
    i           : int       = 0
    stack       : List[int] = []
    entities    : List[Tuple[str,int,int]] = []

    while i < len(text):
        if text.startswith("##", i):
            if _is_open(text, i, inside=bool(stack)):
                stack.append(clean_pos)                     # OPEN
            elif stack:                                     # CLOSE
                start = stack.pop()
                entities.append(("".join(clean[start:clean_pos]),
                                   start, clean_pos))
            i += 2
            continue

        clean.append(text[i])                               # copy char
        clean_pos += 1
        i += 1

    # keep outer-most spans only?
    if longest_only:
        outer: List[Tuple[str,int,int]] = []
        for t, s, e in sorted(entities, key=lambda x: x[2]-x[1], reverse=True):
            if not any(s >= os and e <= oe for _, os, oe in outer):
                outer.append((t, s, e))
        return outer

    return entities
# ---------------------------------------------------------------------

# ────────────────────────────────────────────────────────────
# 1.  MARKED-TEXT PARSER  (identical to previous version)
# ────────────────────────────────────────────────────────────


# def extract_entities(marked: str, longest_only: bool = False
#                      ) -> List[Tuple[str, int, int]]:
#     clean, clean_pos, i, stack, ents = [], 0, 0, [], []
#     while i < len(marked):
#         if marked.startswith("##", i):
#             if _is_open(marked, i, bool(stack)):
#                 stack.append(clean_pos)
#             elif stack:
#                 s = stack.pop()
#                 ents.append(("".join(clean[s:clean_pos]), s, clean_pos))
#             i += 2;  continue
#         clean.append(marked[i]);  clean_pos += 1;  i += 1
#
#     if longest_only:
#         outer = []
#         for t,s,e in sorted(ents, key=lambda x: x[2]-x[1], reverse=True):
#             if not any(s>=os and e<=oe for _,os,oe in outer):
#                 outer.append((t,s,e))
#         return outer
#     return ents

# ────────────────────────────────────────────────────────────
# 2.  ALIGNMENT helpers
# ────────────────────────────────────────────────────────────
def _find_next(hay: str, needle: str, start: int) -> Tuple[int,int]:
    k = hay.find(needle, start)
    return (k, k+len(needle)) if k != -1 else (-1, -1)

def _remove_nested(spans: List[Dict]) -> List[Dict]:
    """
    Throw away a span that is fully contained in (begin >=,
    end <=) any longer span already kept.
    """
    keep: List[Dict] = []
    # longest → shortest ensures outer spans are kept first
    for sp in sorted(spans, key=lambda d: d['end']-d['begin'], reverse=True):
        if not any(sp['begin'] >= k['begin'] and sp['end'] <= k['end']
                   for k in keep):
            keep.append(sp)
    # return them in document order
    return sorted(keep, key=lambda d: d['begin'])

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
                out.append({"text": text, "begin": m.start(), "end": m.end()})
        else:
            b,e = _find_next(original, text, cur)
            if b != -1:
                out.append({"text": text, "begin": b, "end": e})
                cur = e

    return _remove_nested(out)

if __name__ == "__main__":
    # example = ("####Claremore## Lake## is a reservoir in ##Rogers County##, ##Oklahoma##. Constructed in 1929-1930 by damming ##Dog Creek## for the purpose of providing water to the city of ##Claremore##, ##Oklahoma## and houses recreational amenities such as boat ramps, fishing docks, and picnic areas. In 2011, the lake added a 9-hole disc golf course.",
    #            "A further international service is provided by ##Venice Simplon Orient Express##. Although this is primarily a railtour operator, with special trains to various locations in the ##United Kingdom##, it also operates the scheduled ##Orient Express## service to destinations in ##Europe##. This involves two separate trains; the ##Belmond British Pullman## departs from ##London Victoria## and terminates at ##Folkestone West##, where passengers transfer by coach through the ##Channel Tunnel## to ##Calais##; at ##Gare de ##Calais####, they then join the ##Orient Express## which then calls at various destinations including ##Paris##, ##Vienna##, ##Innsbruck##, ##Venice## and ##Rome##.",
    #            "##Siem Reap## is a cluster of small villages along the ####Siem Reap## River##. These villages were originally developed around Buddhist pagodas (wats) which are almost evenly spaced along the river from ##Wat Preah En Kau Sei## in the north to ##Wat Phnom Krom## in the south, where the ####Siem Reap## River## meets the great ##Tonlé Sap Lake##.",
    #            "The production of wine declined significantly after the ##Muslim conquest of ##Egypt#### in the 7th century. Attitudes towards alcohol varied greatly under Islamic rule, Muslim rulers generally showed some level of tolerance towards alcohol production controlled by religious minorities. ##Jewish## manuscripts from the ##Cairo Geniza## recount the involvement of ##Egyptian## ##Jewish##s in the production and sale of wine in medieval ##Egypt##. The consumption of wine was not necessarily limited to religious minorities however. ##Western## travelers and pilgrims passing through ##Cairo## on their journeys reported that ##Muslim## locals imbibed on wine and a local barley beer, known as ##booza## (, not to be confused with the ##Levantine## ice cream of the same name), even during the most draconian periods of Islamic rule. The most popular wine was known as ##nebit shamsi## (), made from imported raisins and honey and left to ferment in the sun (hence the name, which roughly translates into sun wine).",
    #            "1992: ####South Australia#### 19.19.(133) d ##Victoria## 18.12.(120). ##Wayne Carey## (SA) described this game as the reason he believed he could succeed in the ##AFL##. In a high scoring game, ##Stephen Kernahan## (SA) kicked six goals, ##Paul Salmon## (Vic) kicked five and ##Paul Roos## (Vic) kicked three. ##Wayne Carey## dominated at centre half forward and kicked two goals. ####South Australia#### won the game in the final moments.")
    #

    json_path = '/Users/orsh/.clearml/cache/storage_manager/datasets/ds_a5889b5f80164cd49ae3c149af5366f1/nertrieve_span_extraction.json'
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for record in data.values():
        generated_text = record["generated"]
        original_text = record["sentence"]
        print(original_text)
        print(generated_text)
        for ent in align_to_original(marked = generated_text, original=original_text, longest_only=True,all_occurrences=True):
            print(ent)
            print(f"----{original_text[ent['begin']:ent['end']]}----")
        print("\n" + "=" * 80 + "\n")