// ws_review.js — lớp xem dự đoán của model, CHỈ ĐỌC.
// =================== lớp xem dự đoán của model (CHỈ ĐỌC) ===================
// Dữ liệu lấy thẳng từ runs/eval/ qua /api/pred/*, không có bản sao nào trong
// data/. Lớp này KHÔNG BAO GIỜ đi vào `inst`: app autosave sau 700 ms và còn
// bắn sendBeacon lúc rời trang, nên một dự đoán lọt vào `inst` là bị ghi thẳng
// thành nhãn thật trong masks/corrected/, im lặng, cho cả 850 ảnh.
let predOn=false,predRuns=[],predRun='',predConf=0.5,predData=null,predShowGt=true;
let predToken=0,predTimer=null,predHover=null,predBefore='sam-click';
function predColors(){const s=getComputedStyle(document.documentElement);
 return {tp:(s.getPropertyValue('--pred-tp')||'#22d3ee').trim(),
         fp:(s.getPropertyValue('--pred-fp')||'#e879f9').trim(),
         fn:(s.getPropertyValue('--pred-fn')||'#fb7185').trim()};}
// Bề rộng trên MÀN HÌNH của một polygon — dùng để quyết định có viết chữ lên
// nó không. Tán trung vị rộng 328 px ảnh, ở mức khớp khung (~0,43) là ~141 px,
// đủ chỗ; nhưng lúc zoom ra thì không, và chữ chồng chữ khó đọc hơn không chữ.
function polyW(p){let a=Infinity,b=-Infinity;
 for(let k=0;k<p.length;k+=2){if(p[k]<a)a=p[k];if(p[k]>b)b=p[k];}
 return (b-a)*scale;}
// Vẽ SAU cùng, đè lên nhãn. Ba quy ước, chọn theo số đo chứ không theo cảm tính:
//
// 1. Dự đoán CHỈ CÓ VIỀN, nhãn mới có nền. Hai đường biên của một cặp đã ghép
//    cách nhau trung vị 12,5-13,5 px ảnh (sáu model, 11,5k cặp) = 5,4-5,8 px
//    màn hình ở mức khớp khung. Tô nền cả hai lớp là nuốt trọn đúng dải sai
//    lệch mà người ta mở lớp này lên để nhìn.
// 2. Viền dự đoán là NÉT ĐỨT. Chỗ hai đường trùng nhau (sai lệch dưới 1 px
//    màn hình lúc zoom ra), nét đứt để đường liền của nhãn lộ ra qua khe —
//    khác màu thôi thì đường vẽ sau che hẳn đường vẽ trước.
// 3. Chỉ TÔ NỀN cho thứ SAI: dự đoán thừa (hồng) và tán bị bỏ sót (đỏ). Nền
//    ở đây nghĩa là "có chuyện", viền trơn nghĩa là "khớp".
function drawPred(){
 if(!predOn||!predData||!predData.ok)return;
 const C=predColors();
 ctx.save();
 // Tán bị BỎ SÓT — thứ đáng nhìn nhất trong một lượt so sánh, nên phải là thứ
 // nổi nhất. Vẽ VIỀN TỐI rộng phía dưới rồi mới tô màu lên: nền ruộng lúc
 // trưa sáng gần trắng, màu hồng nhạt đặt thẳng lên đó thì chìm hẳn (đã thử).
 // Nét dài hơn nét của dự đoán để hai lớp không lẫn nhau dù cùng là nét đứt.
 (predData.gt||[]).forEach(g=>{if(g.matched||!g.points||g.points.length<6)return;
  polyPath(g.points);
  ctx.fillStyle=hexA(C.fn,0.22);ctx.fill();
  ctx.setLineDash([]);ctx.lineWidth=6;ctx.strokeStyle='rgba(6,12,22,.55)';ctx.stroke();
  ctx.setLineDash([10,5]);ctx.lineWidth=3;ctx.strokeStyle=C.fn;ctx.stroke();});
 ctx.setLineDash([6,4]);
 // Một dự đoán rời thành nhiều vòng vẫn là MỘT dự đoán: đếm theo `det` để
 // ngưỡng "quá nhiều chữ" không bị tỉ lệ vụn của model làm lệch, và chỉ vòng
 // LỚN NHẤT của mỗi dự đoán mới được viết chữ (nếu không, một tán vỡ ba mảnh
 // hiện ba lần cùng một con số).
 const lon={};
 (predData.polygons||[]).forEach((q,i)=>{const w=polyW(q.points);
  if(lon[q.det]===undefined||w>lon[q.det].w)lon[q.det]={i:i,w:w};});
 (predData.polygons||[]).forEach((q,i)=>{const p=q.points;if(!p||p.length<6)return;
  const hov=(predHover&&predHover.kind==='pred'&&predHover.i===i);
  polyPath(p);
  if(!q.matched){ctx.fillStyle=hexA(C.fp,hov?0.26:0.12);ctx.fill();}
  else if(hov){ctx.fillStyle=hexA(C.tp,0.18);ctx.fill();}
  ctx.strokeStyle=q.matched?C.tp:C.fp;ctx.lineWidth=hov?3.5:(q.matched?2:2.5);ctx.stroke();});
 // Chữ vẽ thành lượt RIÊNG để nó không bị viền của tán vẽ sau đè lên.
 //
 // Không chặn bằng "quá N dự đoán thì thôi": ngưỡng đếm vừa chặn nhầm lúc
 // zoom sâu (41 tán to đùng vẫn thừa chỗ) vừa không cứu được lúc SOLOv2 xếp
 // 100 dự đoán trùng nhau lên cùng một tán. Vấn đề thật là CHỮ ĐÈ CHỮ, nên
 // đo đúng thứ đó: giữ lại các khung chữ đã vẽ và bỏ qua khung nào chạm vào
 // chúng. Dự đoán điểm cao được xét trước (danh sách đã sắp theo điểm) nên
 // chỗ chật thì con số đáng tin nhất là con số còn lại.
 ctx.setLineDash([]);ctx.font='600 11px ui-monospace,Menlo,Consolas,monospace';
 ctx.textAlign='center';ctx.textBaseline='middle';
 const da=[];
 (predData.polygons||[]).forEach((q,i)=>{const p=q.points;if(!p||p.length<6)return;
  if(lon[q.det].i!==i||polyW(p)<54)return;
  const c=centroid(p),x=sx(c[0]),y=sy(c[1]);
  const t=q.conf.toFixed(2)+(q.matched?(' · '+q.iou.toFixed(2)):' ✗');
  const w=ctx.measureText(t).width+8,L=x-w/2,R=x+w/2,T=y-8,B=y+8;
  if(da.some(r=>L<r[2]&&R>r[0]&&T<r[3]&&B>r[1]))return;
  da.push([L,T,R,B]);
  ctx.fillStyle='rgba(6,12,22,.72)';ctx.fillRect(L,T,w,16);
  ctx.fillStyle=q.matched?C.tp:C.fp;ctx.fillText(t,x,y);});
 ctx.textAlign='start';ctx.textBaseline='alphabetic';
 ctx.restore();}
// Tra cứu bằng chuột ở chế độ xem: dự đoán được ưu tiên vì nó nằm TRÊN.
function predPick(x,y){
 if(!predOn||!predData||!predData.ok)return null;
 const ps=predData.polygons||[];
 for(let i=ps.length-1;i>=0;i--)if(ps[i].points.length>=6&&pip(ps[i].points,x,y))
  return {kind:'pred',i:i,r:ps[i]};
 const gs=predData.gt||[];
 for(let i=gs.length-1;i>=0;i--)if(gs[i].points&&gs[i].points.length>=6&&pip(gs[i].points,x,y))
  return {kind:'gt',i:i,r:gs[i]};
 return null;}
function predHint(){
 if(!predData||!predData.ok){$('hint').textContent=HINT.view;return;}
 const h=predHover;
 if(!h){$('hint').textContent=HINT.view;return;}
 $('hint').textContent=(h.kind==='pred')
  ? ('Dự đoán · điểm '+h.r.conf.toFixed(3)+(h.r.matched
      ? (' · ghép với nhãn #'+(h.r.gt+1)+' ở IoU '+h.r.iou.toFixed(3))
      : ' · KHÔNG ghép được nhãn nào (dự đoán thừa)'))
  : ('Nhãn #'+(h.i+1)+' · '+(h.r.matched
      ? ('được ghép ở IoU '+h.r.iou.toFixed(3))
      : 'KHÔNG model nào ghép được (bỏ sót)'));}
// --- khoá sửa ---------------------------------------------------------------
// Một hàm duy nhất, gọi ở MỌI lối vào có thể đổi hoặc lưu nhãn tán. Chế độ
// 'view' (Xem của thẻ Tán, và lúc thẻ So sánh đang mở) làm bốn nhánh sửa trong
// handler chuột không khớp nữa; các nút trong danh sách tán và phím tắt đi
// đường riêng nên vẫn phải hỏi ở đây.
function editLocked(){return predOn||mode==='view';}
// --- nạp dữ liệu ------------------------------------------------------------
// `tok` là selToken lúc bắt đầu. Danh sách lượt chấm phụ thuộc ẢNH (mỗi ảnh
// nằm trong split test của đúng một fold), nên hai lần đổi ảnh chồng nhau mà
// không chốt thì lượt về sau có thể ghi đè bằng kết quả của ảnh trước — đúng
// lý do `selectImg` đã có selToken.
async function predLoadRuns(tok){
 predRuns=[];predRun='';
 const sel=$('predRun');sel.innerHTML='';
 if(!current){$('predCount').textContent='—';return;}
 const ten=current;
 let d=null;
 try{d=await(await fetch('/api/pred/runs?name='+enc(ten),{cache:'no-store'})).json();}catch(e){d=null;}
 if(ten!==current||(tok!==undefined&&tok!==selToken))return;
 predRuns=(d&&d.runs)||[];
 $('predCount').textContent=predRuns.length?(predRuns.length+' lượt'):'0';
 if(!predRuns.length){
  const o=document.createElement('option');
  o.textContent=(d&&d.error)?d.error:'Không có lượt chấm nào phủ ảnh này';
  o.value='';sel.appendChild(o);sel.disabled=true;return;}
 sel.disabled=false;
 predRuns.forEach(r=>{const o=document.createElement('option');o.value=r.run;o.textContent=r.label;
  o.title=r.run+'\n'+(r.weights||'');sel.appendChild(o);});
 // Giữ nguyên model đang xem khi đổi ảnh, nếu ảnh mới cũng có model đó.
 const giu=predRuns.find(r=>r.model===(predData&&predData.source&&predData.source.model));
 predRun=(giu||predRuns[0]).run;sel.value=predRun;}
async function predFetch(){
 if(!predOn||!current||!predRun){predData=null;predRender();draw();return;}
 const tok=++predToken,ten=current;
 $('predSum').innerHTML='<span class="pchip">đang đọc…</span>';
 let d=null;
 try{d=await(await fetch('/api/pred/data?name='+enc(ten)+'&run='+encodeURIComponent(predRun)
   +'&conf='+predConf,{cache:'no-store'})).json();}catch(e){d={ok:false,error:String(e)};}
 // Hai chốt khác nhau: `predToken` bắt việc đổi model/ngưỡng giữa chừng, còn
 // so tên ảnh bắt việc đổi ẢNH giữa chừng — vẽ dự đoán của ảnh khác lên ảnh
 // đang mở là sai lặng lẽ, nhìn vẫn ra polygon nên không ai nghi.
 if(tok!==predToken||ten!==current)return;
 predData=d;predHover=null;predRender();draw();predHint();}
function predSchedule(){clearTimeout(predTimer);predTimer=setTimeout(predFetch,180);}
function predRender(){
 const box=$('predSum'),src=$('predSrc'),w=$('predWarn');
 if(!predData){box.innerHTML='';src.textContent='';w.hidden=true;renderNeedLook();return;}
 if(!predData.ok){box.innerHTML='<span class="pchip fp">'+(predData.error||'lỗi')+'</span>';
  src.textContent='';w.hidden=true;renderNeedLook();return;}
 const s=predData.summary;
 box.innerHTML=
  '<span class="pchip tp">khớp <b>'+s.tp+'</b></span>'+
  '<span class="pchip fp">thừa <b>'+s.fp+'</b></span>'+
  '<span class="pchip fn">sót <b>'+s.fn+'</b></span>'+
  '<span class="pchip">IoU TB <b>'+(s.mean_iou==null?'—':s.mean_iou.toFixed(3))+'</b></span>'+
  '<span class="pchip">'+s.n_pred+'/'+s.n_pred_all+' dự đoán · '+s.n_gt+' nhãn</span>';
 const o=predData.source;
 src.textContent=o.run+'  ·  imgsz '+(o.imgsz||'?')+'  ·  chấm ở conf '+(o.conf==null?'?':o.conf)
  +'  ·  ghép ở IoU '+predData.iou_thr+'\n'+(o.weights||'');
 w.hidden=!predData.warn;w.textContent=predData.warn||'';renderNeedLook();}
async function predOnImage(tok){
 if(!predOn)return;
 await predLoadRuns(tok);
 if(tok!==undefined&&tok!==selToken)return;
 await predFetch();}
wsRegister({id:'review',
 // Đổi ảnh thì lớp dự đoán phải đổi theo: mỗi ảnh nằm trong split test của
 // đúng một fold, nên danh sách lượt chấm cũng khác.
 onImage(t,tok){predData=null;predHover=null;predRender();predOnImage(tok);},
 draw(){drawPred();}});
// Thẻ So sánh = lớp dự đoán đang bật, chỉ đọc theo cấu tạo. Vào thẻ thì bật
// (lưu nốt phần đang sửa TRƯỚC khi khoá), rời thẻ thì tắt và trả thẻ Tán về
// đúng chế độ cũ, nên ở thẻ Tán không còn dự đoán nào che nhãn.
async function setPredOn(on){if(predOn===on)return;predOn=on;
 if(on){flush();predBefore=mode;
  sel=-1;dragPt=-1;hovPt=-1;ghost=null;draft=[];boxStart=null;
  setMode('view');await predLoadRuns(selToken);await predFetch();}
 else{predData=null;predHover=null;setMode(predBefore);predRender();draw();}}
// "Cần xem": tán bị sót và dự đoán thừa ở ngưỡng điểm đang chọn; bấm là phóng
// tới. Dự đoán rời thành nhiều vòng chỉ tính một lần, theo vòng lớn nhất.
function renderNeedLook(){const box=$('needLook');if(!box)return;box.innerHTML='';
 if(!predOn||!predData||!predData.ok){box.innerHTML='<span class="muted">—</span>';return;}
 const items=[];
 (predData.gt||[]).forEach((g,i)=>{if(!g.matched&&g.points&&g.points.length>=6)
  items.push({kind:'gt',i:i,pts:g.points,txt:'Sót · tán #'+(i+1)+': không dự đoán nào ghép được'});});
 const big={};
 (predData.polygons||[]).forEach((q,i)=>{if(q.matched||!q.points||q.points.length<6)return;
  const w=polyW(q.points);if(!big[q.det]||w>big[q.det].w)big[q.det]={i:i,w:w};});
 Object.values(big).map(b=>predData.polygons[b.i]&&{i:b.i,q:predData.polygons[b.i]})
  .sort((a,b)=>b.q.conf-a.q.conf)
  .forEach(o=>items.push({kind:'pred',i:o.i,pts:o.q.points,txt:'Thừa · dự đoán điểm '+o.q.conf.toFixed(2).replace('.',',')}));
 if(!items.length){box.innerHTML='<span class="muted">Không có tán sót hay dự đoán thừa ở ngưỡng này.</span>';return;}
 items.forEach(it=>{const b=document.createElement('button');b.className='btn needitem '+(it.kind==='gt'?'fn':'fp');b.textContent=it.txt;
  b.onclick=()=>{zoomToPoly(it.pts);predHover={kind:it.kind,i:it.i,r:it.kind==='gt'?predData.gt[it.i]:predData.polygons[it.i]};predHint();draw();};
  box.appendChild(b);});}
tabRegister({id:'review',inspTitle:'Kết quả trên ảnh',
 help:[['Rê chuột lên một vùng','Xem điểm và IoU của dự đoán hoặc nhãn'],['Bấm một mục Cần xem','Phóng tới tán sót hoặc dự đoán thừa đó'],
  ['Thanh Điểm ≥','Đổi ngưỡng điểm của dự đoán'],['Hiện cả nhãn tay','Bật/tắt nền nhãn để nhìn rõ dự đoán']],
 enter(){setPredOn(true);},leave(){setPredOn(false);}});
