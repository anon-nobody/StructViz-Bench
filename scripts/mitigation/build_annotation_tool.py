#!/usr/bin/env python3
"""Build the single-file human-annotation tool (human_eval_package/annotate.html).

The tool shows image + question + ground truth on one screen, records
Correct / Ambiguous / Incorrect with one click (or keys 1/2/3), autosaves progress per
annotator in the browser (localStorage), and exports ratings_annotatorN.csv with EXACTLY the
header and row order of the blank sheets, so assemble_ratings.py / human_eval_aggregate.py run
unchanged. Works offline from file:// (images are referenced relatively; no fetch is used).

Usage:
  python scripts/mitigation/build_annotation_tool.py            # writes human_eval_package/annotate.html
"""
from __future__ import annotations
import csv, json, os, re, sys
import os

ROOT = os.environ.get("STRUCTVIZ_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PKG = os.path.join(ROOT, "human_eval_package")
HEADER = ["item_id", "modality", "task", "viz_type", "image", "question",
          "ground_truth", "rating", "your_answer", "notes"]


# --- Korean glosses for the question templates -------------------------------------------
# The English question is authoritative (it is exactly what the model was shown); the Korean
# line is only a reading aid for annotators. {0},{1},... are the quoted column names and
# numbers captured from the English question, in the order they appear there.
PLACEHOLDER_RE = re.compile(r"'[^']*'|\b\d+(?:\.\d+)?\b")

KO = {
    "If all values in '<X>' were multiplied by <N>, would '<X>' remain above '<X>' on average?":
        "'{0}'의 모든 값을 {1}배 하면, 평균적으로 '{2}' 쪽이 '{3}'보다 계속 높을까요?",
    'Which node has the highest betweenness centrality?':
        '매개 중심성(betweenness centrality)이 가장 높은 노드는?',
    "Which '<X>' has the highest '<X>' value?":
        "'{0}' 중 '{1}' 값이 가장 높은 항목은?",
    'If every value increased by <N>%, would max index change?':
        '모든 값이 {0}% 증가하면, 최댓값의 위치(인덱스)가 바뀔까요?',
    "How many rows have '<X>' strictly greater than <N>?":
        "'{0}' 값이 {1}보다 엄격히 큰 행은 몇 개인가요?",
    "Which '<X>' entry deviates most from the mean of '<X>'?":
        "'{1}' 평균에서 가장 많이 벗어난 '{0}' 항목은?",
    "Which '<X>' has the lowest '<X>' value?":
        "'{0}' 중 '{1}' 값이 가장 낮은 항목은?",
    'Is there a path between node <N> and node <N>?':
        '노드 {0}, 노드 {1} 사이에 경로가 있나요?',
    'Does the peak occur after the trough?':
        '고점(peak)이 저점(trough)보다 뒤에 나타나나요?',
    'How many immediate neighbors does node <N> have?':
        '노드 {0}의 직접 이웃은 몇 개인가요?',
    'How many nodes are in the largest connected component?':
        '가장 큰 연결 요소(connected component)에 포함된 노드는 몇 개인가요?',
    "Do '<X>' and '<X>' appear positively or negatively correlated?":
        "'{0}', '{1}' 두 열은 양(+)의 상관으로 보이나요, 음(−)의 상관으로 보이나요?",
    'If we add <N> only to second half, would second-half mean exceed first-half mean?':
        '후반부에만 {0}만큼 더하면, 후반부 평균이 전반부 평균을 넘을까요?',
    "What is the value of '<X>' in row <N>?":
        "{1}번 행의 '{0}' 값은 무엇인가요?",
    'What is the average degree among node <N>, node <N>, and node <N>?':
        '노드 {0}, 노드 {1}, 노드 {2}의 평균 차수(degree)는?',
    'Is normalized amplitude (range/|mean|) greater than <N>?':
        '정규화 진폭(범위/|평균|)이 {0}보다 큰가요?',
    'What is the range (max - min) of values?':
        '값들의 범위(최댓값 − 최솟값)는?',
    'Is the median value above or below the overall mean?':
        '중앙값이 전체 평균보다 위인가요, 아래인가요?',
    'If the edge between node <N> and node <N> were removed, would the number of connected components increase?':
        '노드 {0}, 노드 {1} 사이의 간선을 제거하면, 연결 요소의 개수가 늘어날까요?',
    'How many sign changes are in first-order differences?':
        '1차 차분에서 부호가 바뀌는 횟수는 몇 번인가요?',
    "What is the sum of '<X>' across all rows? Round to <N> decimals.":
        "모든 행에 걸친 '{0}'의 합은? 소수점 {1}자리로 반올림하세요.",
    'Is this series more volatile in the first or second half?':
        '이 시계열은 전반부와 후반부 중 어느 쪽이 더 변동성이 큰가요?',
    "What is the median of '<X>' across all rows? Round to <N> decimals.":
        "모든 행에 걸친 '{0}'의 중앙값은? 소수점 {1}자리로 반올림하세요.",
    'Are node <N> and node <N> connected?':
        '노드 {0}, 노드 {1}은 서로 연결되어 있나요?',
    'Which half has larger mean absolute first difference?':
        '1차 차분 절댓값의 평균이 더 큰 쪽은 전반부인가요, 후반부인가요?',
    'Which node has the highest degree?':
        '차수(degree)가 가장 높은 노드는?',
    'Is this graph bipartite?':
        '이 그래프는 이분 그래프(bipartite)인가요?',
    'If an edge were added between node <N> and node <N>, would they be directly connected?':
        '노드 {0}, 노드 {1} 사이에 간선을 추가하면, 두 노드가 직접 연결될까요?',
    'If one random edge were removed, would the total edge count decrease by one?':
        '임의의 간선 하나를 제거하면, 전체 간선 수가 1만큼 줄어들까요?',
    'At approximately which timestep does a significant change occur?':
        '대략 몇 번째 시점(timestep)에서 유의미한 변화가 일어나나요?',
    "Which column has the higher average value: '<X>' or '<X>'?":
        "'{0}', '{1}' 중 평균값이 더 높은 열은?",
    'What is the pattern type label for this dataset?':
        '이 데이터셋의 패턴 유형 레이블은 무엇인가요?',
    'What is the first value in the series?':
        '이 시계열의 첫 번째 값은?',
    'Does this graph contain a cycle?':
        '이 그래프에 사이클(cycle)이 있나요?',
    "Is '<X>' generally increasing or decreasing across rows?":
        "'{0}' 열은 행을 따라 전반적으로 증가하나요, 감소하나요?",
    'Based on the most recent change, what is the next expected value?':
        '가장 최근의 변화를 근거로, 다음에 예상되는 값은?',
    'What is the shortest path length between node <N> and node <N>?':
        '노드 {0}, 노드 {1} 사이의 최단 경로 길이는?',
    'What is the degree of node <N>?':
        '노드 {0}의 차수(degree)는?',
    'What is the average degree of the graph?':
        '이 그래프의 평균 차수(degree)는?',
    'What is the last value in the series?':
        '이 시계열의 마지막 값은?',
    'Do both halves move in the same trend direction?':
        '전반부와 후반부가 같은 추세 방향으로 움직이나요?',
    'If the series were multiplied by -<N>, would max and min indices swap?':
        '시계열 전체를 −{0}배 하면, 최댓값과 최솟값의 위치(인덱스)가 서로 바뀔까요?',
    'By how much does second-half mean exceed first-half mean?':
        '후반부 평균은 전반부 평균을 얼마나 초과하나요?',
    'What is the diameter of the graph (or largest component)?':
        '이 그래프(또는 가장 큰 연결 요소)의 지름(diameter)은?',
    'Does this series show periodic/seasonal behavior?':
        '이 시계열은 주기적/계절적 양상을 보이나요?',
    "What is the average of '<X>' across all rows? Round to <N> decimals.":
        "모든 행에 걸친 '{0}'의 평균은? 소수점 {1}자리로 반올림하세요.",
    'Which timestep appears most anomalous relative to the sequence average?':
        '시퀀스 평균에 비추어 가장 이상해 보이는 시점(timestep)은?',
}


def gloss(q: str) -> str:
    """Korean gloss for one question, or '' if its template is unknown."""
    vals = [m.group(0) for m in PLACEHOLDER_RE.finditer(q)]
    tpl = PLACEHOLDER_RE.sub(lambda m: "'<X>'" if m.group(0).startswith("'") else "<N>", q)
    ko = KO.get(tpl)
    if ko is None:
        return ""
    plain = [v[1:-1] if v.startswith("'") else v for v in vals]
    try:
        return ko.format(*plain)
    except (IndexError, KeyError):
        return ""


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StructViz-Bench 인간 검증 (Task A)</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c1e21;--mut:#6b7280;--line:#e5e7eb;--ok:#15803d;--amb:#b45309;--bad:#b91c1c;--acc:#2563eb}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,"Segoe UI",Roboto,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;background:var(--bg);color:var(--ink)}
header{background:#fff;border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0;font-weight:700}
.sel{display:flex;gap:8px;align-items:center}
select,input,textarea,button{font:inherit}
select{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}
.prog{flex:1;min-width:220px;display:flex;align-items:center;gap:10px;color:var(--mut);font-size:13px}
.bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc);width:0}
main{max-width:1180px;margin:16px auto;padding:0 16px;display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.9fr);gap:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.imgwrap{background:#fafafa;border:1px solid var(--line);border-radius:8px;display:flex;align-items:center;justify-content:center;min-height:360px;cursor:zoom-in;overflow:hidden}
.imgwrap img{max-width:100%;max-height:72vh;display:block}
.meta{color:var(--mut);font-size:12.5px;margin-top:8px;display:flex;gap:12px;flex-wrap:wrap}
.q{font-size:17px;font-weight:600;margin:0 0 2px}
.qko{font-size:14px;color:var(--mut);margin:0 0 10px;padding-left:9px;border-left:3px solid var(--line);line-height:1.45}
.gt{font-size:15px;margin:0 0 12px}.gt b{background:#eef2ff;border:1px solid #c7d2fe;border-radius:6px;padding:2px 8px;font-family:ui-monospace,Menlo,Consolas,monospace}
.rate{display:grid;gap:8px;margin:10px 0}
.rate button{padding:12px 12px;border-radius:8px;border:2px solid var(--line);background:#fff;text-align:left;cursor:pointer;line-height:1.35}
.rate button b{display:block;font-size:15px}.rate button span{color:var(--mut);font-size:12.5px}
.rate button.ok.on{border-color:var(--ok);background:#f0fdf4}.rate button.amb.on{border-color:var(--amb);background:#fffbeb}.rate button.bad.on{border-color:var(--bad);background:#fef2f2}
label{display:block;font-size:12.5px;color:var(--mut);margin:10px 0 4px}
input[type=text],textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:#fff}
textarea{min-height:64px;resize:vertical}
.nav{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.nav button,.tools button{padding:9px 12px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
.nav button.pri,.tools button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
.tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}
.note{font-size:12.5px;color:var(--mut);margin-top:8px}
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 12px;font-size:13px;margin-bottom:12px}
.hidden{display:none!important}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(34px,1fr));gap:4px;margin-top:10px}
.grid button{padding:5px 0;font-size:11px;border-radius:5px;border:1px solid var(--line);background:#fff;cursor:pointer}
.grid button.done{background:#e5e7eb}.grid button.cur{outline:2px solid var(--acc)}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px;background:#f9fafb}
</style></head><body>
<header>
 <h1>StructViz-Bench 인간 검증 · Task A (정답 판독성)</h1>
 <div class="sel"><label style="margin:0">평가자</label>
  <select id="ann"><option value="">선택…</option><option value="1">Annotator 1</option><option value="2">Annotator 2</option><option value="3">Annotator 3</option></select></div>
 <div class="prog"><span id="ptxt">0 / 0</span><div class="bar"><i id="pbar"></i></div></div>
</header>
<main>
 <section class="card">
  <div id="gate" class="warn">먼저 상단에서 <b>본인 평가자 번호</b>를 선택하세요. 진행 상황은 이 브라우저에 자동 저장되며, 같은 브라우저에서 이어서 할 수 있습니다.</div>
  <div id="work" class="hidden">
   <div class="imgwrap" id="imgwrap" title="클릭하면 원본 크기로 새 탭에서 열립니다"><img id="img" alt=""></div>
   <div class="meta"><span id="mid"></span><span id="mmod"></span><span id="mviz"></span><span id="mtask"></span></div>
   <div class="grid" id="grid"></div>
  </div>
 </section>
 <aside class="card" id="side">
  <p class="q" id="q">—</p>
  <p class="qko" id="qko" hidden></p>
  <p class="gt">제시된 정답: <b id="gt">—</b></p>
  <div class="rate">
   <button class="ok" data-r="Correct"><b>Correct <kbd>1</kbd></b><span>이미지만 보고 ground_truth가 <u>명확히 맞다</u>.</span></button>
   <button class="amb" data-r="Ambiguous"><b>Ambiguous <kbd>2</kbd></b><span>답은 방어 가능하지만 이미지/질문이 <u>다른 해석도 허용</u>한다.</span></button>
   <button class="bad" data-r="Incorrect"><b>Incorrect <kbd>3</kbd></b><span>ground_truth가 <u>틀려 보인다</u>.</span></button>
  </div>
  <label>your_answer (선택) — 이미지만 보고 <b>본인이 낸 답</b>. 채우면 인간 정확도 상한도 계산됩니다.</label>
  <input type="text" id="ya" placeholder="예: 42 / A / 2021-03">
  <label>notes — Ambiguous/Incorrect를 고른 경우 간단한 이유</label>
  <textarea id="notes" placeholder="예: 축 눈금이 겹쳐 값 구분 불가"></textarea>
  <div class="nav">
   <button id="prev">◀ 이전 <kbd>←</kbd></button>
   <button id="next" class="pri">다음 ▶ <kbd>→</kbd></button>
   <button id="unrated">다음 미판정 항목</button>
  </div>
  <div class="tools">
   <button id="export" class="pri">CSV 내보내기 (제출용)</button>
   <span class="note" id="etxt"></span>
  </div>
  <p class="note">질문은 <b>영어 원문이 기준</b>입니다(모델에게 제시된 그대로). 회색 한국어 줄은 이해를 돕는 참고 번역입니다.</p>
  <p class="note">다른 평가자와 <b>상의하지 마세요</b> — 독립 판정이어야 합의율·Fleiss κ가 유효합니다. 내보낸 <code>ratings_annotatorN.csv</code> 파일을 회신해 주세요.</p>
 </aside>
</main>
<script>
const ITEMS = __ITEMS_JSON__;
const HEADER = __HEADER_JSON__;
const $ = id => document.getElementById(id);
let ann = '', state = {}, idx = 0;
const key = () => 'structviz_taskA_ann' + ann;
function load(){ try{ state = JSON.parse(localStorage.getItem(key())||'{}')||{}; }catch(e){ state={}; } }
function save(){ try{ localStorage.setItem(key(), JSON.stringify(state)); }catch(e){} }
function rec(it){ return state[it.item_id] || (state[it.item_id] = {rating:'', your_answer:'', notes:''}); }
function ratedCount(){ return ITEMS.filter(it => (state[it.item_id]||{}).rating).length; }
function render(){
  const it = ITEMS[idx]; if(!it) return;
  const r = state[it.item_id] || {};
  $('img').src = it.image; $('img').alt = it.item_id;
  $('mid').textContent = it.item_id + '  (' + (idx+1) + '/' + ITEMS.length + ')';
  $('mmod').textContent = 'modality: ' + it.modality; $('mviz').textContent = 'format: ' + it.viz_type;
  $('mtask').textContent = it.task ? 'task: ' + it.task : '';
  $('q').textContent = it.question;
  const ko = it.question_ko || ''; $('qko').textContent = ko; $('qko').hidden = !ko;
  $('gt').textContent = it.ground_truth;
  document.querySelectorAll('.rate button').forEach(b => b.classList.toggle('on', b.dataset.r === r.rating));
  $('ya').value = r.your_answer || ''; $('notes').value = r.notes || '';
  const n = ratedCount(); $('ptxt').textContent = n + ' / ' + ITEMS.length + ' 판정'; $('pbar').style.width = (100*n/ITEMS.length) + '%';
  $('etxt').textContent = n === ITEMS.length ? '모두 판정 완료 — 내보내기 하세요.' : (ITEMS.length - n) + '개 미판정';
  [...$('grid').children].forEach((b,i) => { b.classList.toggle('done', !!(state[ITEMS[i].item_id]||{}).rating); b.classList.toggle('cur', i===idx); });
}
function go(i){ idx = Math.max(0, Math.min(ITEMS.length-1, i)); render(); window.scrollTo({top:0}); }
function rate(v){
  if(!ann) return; const it = ITEMS[idx]; rec(it).rating = v; save(); render();
  if(v === 'Correct'){ setTimeout(() => { if(idx < ITEMS.length-1) go(idx+1); }, 220); }
  else { $('notes').focus(); }
}
function csvCell(v){ v = (v==null?'':String(v)); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g,'""') + '"' : v; }
function exportCSV(){
  if(!ann){ alert('평가자 번호를 먼저 선택하세요.'); return; }
  const missing = ITEMS.length - ratedCount();
  if(missing > 0 && !confirm(missing + '개 항목이 아직 미판정입니다. 그래도 내보낼까요? (미판정은 빈 칸으로 저장됩니다)')) return;
  const rows = [HEADER.join(',')];
  for(const it of ITEMS){ const r = state[it.item_id]||{};
    rows.push([it.item_id, it.modality, it.task||'', it.viz_type, it.image, it.question, it.ground_truth,
               r.rating||'', r.your_answer||'', r.notes||''].map(csvCell).join(',')); }
  const blob = new Blob([rows.join('\n') + '\n'], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'ratings_annotator' + ann + '.csv';
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 800);
}
$('ann').addEventListener('change', e => {
  ann = e.target.value; try{ localStorage.setItem('structviz_taskA_who', ann); }catch(_){}
  if(!ann){ $('work').classList.add('hidden'); $('gate').classList.remove('hidden'); return; }
  load(); $('gate').classList.add('hidden'); $('work').classList.remove('hidden');
  const first = ITEMS.findIndex(it => !(state[it.item_id]||{}).rating); go(first < 0 ? 0 : first);
});
document.querySelectorAll('.rate button').forEach(b => b.addEventListener('click', () => rate(b.dataset.r)));
$('ya').addEventListener('input', e => { if(ann){ rec(ITEMS[idx]).your_answer = e.target.value; save(); } });
$('notes').addEventListener('input', e => { if(ann){ rec(ITEMS[idx]).notes = e.target.value; save(); } });
$('prev').onclick = () => go(idx-1); $('next').onclick = () => go(idx+1);
$('unrated').onclick = () => { const i = ITEMS.findIndex((it,k) => k>idx && !(state[it.item_id]||{}).rating); const j = i<0 ? ITEMS.findIndex(it => !(state[it.item_id]||{}).rating) : i; if(j>=0) go(j); else alert('미판정 항목이 없습니다.'); };
$('export').onclick = exportCSV;
$('imgwrap').onclick = () => { if(ann) window.open($('img').src, '_blank'); };
document.addEventListener('keydown', e => {
  if(!ann || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if(e.key === '1') rate('Correct'); else if(e.key === '2') rate('Ambiguous'); else if(e.key === '3') rate('Incorrect');
  else if(e.key === 'ArrowLeft') go(idx-1); else if(e.key === 'ArrowRight') go(idx+1);
});
ITEMS.forEach((it,i) => { const b = document.createElement('button'); b.textContent = i+1; b.title = it.item_id; b.onclick = () => go(i); $('grid').appendChild(b); });
(function(){ let who=''; try{ who = localStorage.getItem('structviz_taskA_who')||''; }catch(_){} if(who){ $('ann').value = who; $('ann').dispatchEvent(new Event('change')); } })();
</script></body></html>
"""


def main() -> int:
    items = []
    with open(os.path.join(PKG, "items.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    # sanity: same header as the blank sheets, every image present
    with open(os.path.join(PKG, "ratings_annotator1.csv"), newline="") as f:
        sheet_header = next(csv.reader(f))
    if sheet_header != HEADER:
        sys.exit(f"header mismatch: sheet={sheet_header} tool={HEADER}")
    missing = [it["image"] for it in items if not os.path.exists(os.path.join(PKG, it["image"]))]
    if missing:
        sys.exit(f"{len(missing)} images missing, e.g. {missing[:3]}")
    keep = ["item_id", "modality", "task", "viz_type", "question", "ground_truth", "image"]
    slim = [{k: it.get(k, "") for k in keep} for it in items]
    missing = []
    for d in slim:
        d["question_ko"] = gloss(d["question"])
        if not d["question_ko"]:
            missing.append(d["question"])
    if missing:
        uniq = sorted(set(missing))
        sys.exit(f"{len(missing)} questions lack a Korean gloss ({len(uniq)} distinct). "
                 f"Add them to KO. First: {uniq[0]!r}")
    html = (TEMPLATE
            .replace("__ITEMS_JSON__", json.dumps(slim, ensure_ascii=False))
            .replace("__HEADER_JSON__", json.dumps(HEADER)))
    out = os.path.join(PKG, "annotate.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}  ({len(items)} items embedded, {os.path.getsize(out)//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
