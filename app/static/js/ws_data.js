// ws_data.js — xuất bộ dữ liệu và cửa sổ chọn thư mục lưu.
// ============================ XUẤT DỮ LIỆU ============================
// Luồng: chọn phạm vi + định dạng + cách chia -> chọn nơi lưu -> xuất.
// Mọi phạm vi đều CHỈ tính ảnh đã gán nhãn; con số lấy từ /api/export_preview
// để người dùng biết trước sẽ xuất bao nhiêu, thay vì bấm xong mới biết.
const PREV={};
function scopeOf(v){return v==='all'?'':(v==='img'?(current||''):(pathFilter||''));}
function curScope(){const el=document.querySelector('input[name=exsc]:checked');return el?el.value:'dir';}
async function preview(v){const sc=scopeOf(v);
 if(v==='img'&&!current)return null;
 if(PREV[sc]!==undefined)return PREV[sc];
 try{PREV[sc]=await(await fetch('/api/export_preview?scope='+encodeURIComponent(sc),{cache:'no-store'})).json();}
 catch(e){PREV[sc]=null;}
 return PREV[sc];}
function fmtCnt(p){if(!p)return '—';
 // Không vùng nào bị loại lúc xuất nữa: mọi polygon đã vẽ đều là một tán.
 return p.images_labeled+'/'+p.images_total+' ảnh · '+p.annotations+' vùng'+(p.images_empty?' · '+p.images_empty+' ảnh nền':'');}
async function refreshCounts(){
 $('scRowDir').querySelector('.exn').textContent='Thư mục đang chọn'+(pathFilter?(' · '+pathFilter):' · tất cả');
 $('scRowImg').querySelector('.exn').textContent='Ảnh hiện tại'+(current?(' · '+current.split('/').pop()):'');
 $('scRowImg').classList.toggle('dis',!current);
 $('scRowImg').querySelector('input').disabled=!current;
 for(const [v,id] of [['img','scCntImg'],['dir','scCntDir'],['all','scCntAll']]){
  $(id).textContent='…';const p=await preview(v);$(id).textContent=fmtCnt(p);}
 renderFolders(await preview(curScope()));
 syncSplit();}
// Danh sách đường bay lấy từ /api/export_preview chứ không tự dựng ở client:
// khoá nhóm phía server tính theo BASE, nếu client tự ghép đường dẫn sẽ lệch khi
// người dùng đổi thư mục nguồn, và gán nhầm folder mà không ai biết.
const FLD={};
function renderFolders(p){const box=$('fldList');box.innerHTML='';
 const list=(p&&p.folders)||[];
 if(!list.length){box.innerHTML='<div class="fldrow"><span class="fn">(chưa có ảnh đã gán nhãn)</span></div>';return;}
 list.forEach(f=>{
  if(FLD[f.key]===undefined)FLD[f.key]='train';
  const row=document.createElement('div');row.className='fldrow';
  const nm=document.createElement('span');nm.className='fn';nm.textContent=f.key;nm.title=f.key;
  const ct=document.createElement('span');ct.className='fc';ct.textContent=f.images+' ảnh';
  const sel=document.createElement('select');
  ['train','val','test'].forEach(o=>{
   const op=document.createElement('option');op.value=o;op.textContent=o;
   if(FLD[f.key]===o)op.selected=true;sel.appendChild(op);});
  sel.onchange=()=>{FLD[f.key]=sel.value;syncSplit();};
  row.appendChild(nm);row.appendChild(ct);row.appendChild(sel);box.appendChild(row);});}
function foldersOf(kind){return Object.keys(FLD).filter(k=>FLD[k]===kind);}
function syncSplit(){const m=$('spMode').value;
 $('spBox').style.display=(m==='ratio')?'':'none';
 $('fldWrap').style.display=(m==='folder')?'':'none';
 const v=+$('spVal').value||0,t=+$('spTest').value||0;
 $('spTrain').textContent=Math.max(0,100-v-t);
 const n=$('spNote');
 if(m==='none')n.textContent='data.yaml sẽ KHÔNG có khoá val — ultralytics dừng với "\'val:\' key missing" cho tới khi bạn chia. Cố ý như vậy để không ai lỡ đo trên chính tập train.';
 else if(m==='folder'){const nv=foldersOf('val').length,nt=foldersOf('test').length;
  n.textContent=(nv||nt)
   ? 'Gán nguyên đường bay vào từng tập — cách chia duy nhất không rò rỉ. val: '+nv+' đường bay · test: '+nt+'.'
   : '⚠ Chưa gán đường bay nào cho val/test — tất cả sẽ vào train, data.yaml sẽ không khai khoá val.';}
 else if(m==='field')n.textContent='Giữ nguyên field: chỉ tách được ở mức field, không tách được hai đường bay trong cùng một field.';
 else if(v+t>=100)n.textContent='⚠ val + test ≥ 100%, không còn ảnh nào cho train.';
 else n.textContent='⚠ RÒ RỈ: xáo trong từng đường bay sẽ tách các khung liền kề chụp trùng cây ra hai tập (đo được 35/392 cặp liền kề chồng lấn). Điểm val sẽ cao hơn thực tế — dùng "Theo đường bay" nếu cần con số đáng tin.';}
function openExport(){PREV.__x=0;for(const k in PREV)delete PREV[k];
 $('expInfo').textContent='';$('expModal').classList.add('open');refreshCounts();}
async function doExport(body,p){const btn=$('expGo');btn.disabled=true;
 const t0=Date.now();const n=p?p.images_labeled:0;
 const tick=setInterval(()=>{$('expInfo').textContent='⏳ Đang xuất '+n+' ảnh… '+Math.round((Date.now()-t0)/1000)+'s';},250);
 try{flush();
  const r=await(await fetch('/api/export_dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  clearInterval(tick);btn.disabled=false;
  if(r.exists){if(confirm('Thư mục "'+body.name+'" đã tồn tại.\n\nGhi đè toàn bộ nội dung cũ?')){body.overwrite=true;return doExport(body,p);}
   $('expInfo').textContent='⚠ Đã huỷ — thư mục cũ giữ nguyên.';return;}
  if(!r.ok){$('expInfo').textContent='⚠ '+(r.error||'Lỗi xuất');return;}
  const s=r.splits||{};
  const parts=Object.keys(s).filter(k=>s[k].images).map(k=>k+' '+s[k].images);
  const nimg=Object.values(s).reduce((a,b)=>a+b.images,0);
  const nann=Object.values(s).reduce((a,b)=>a+b.annotations,0);
  let msg='✔ '+nimg+' ảnh · '+nann+' vùng'+(parts.length>1?(' ('+parts.join(' · ')+')'):'')
        +' · 1 lớp canopy · '+(r.formats||[]).join(', ')+' → '+r.dir
        +'  ['+Math.round((Date.now()-t0)/1000)+'s]';
  if((r.problems||[]).length)msg+='\n⚠ '+r.problems.length+' ảnh có vấn đề: '+r.problems.slice(0,3).join(' · ');
  $('expInfo').textContent=msg;}
 catch(e){clearInterval(tick);btn.disabled=false;$('expInfo').textContent='⚠ '+e;}}
// Xuất một ảnh lẻ giờ là mức phạm vi "Ảnh hiện tại" trong hộp thoại Xuất data,
// ghi ra thư mục như mọi phạm vi khác. Pascal VOC đã bỏ: nó chỉ lưu khung chữ
// nhật bao quanh, vứt sạch polygon nên vô dụng với bài toán phân đoạn tán.
let fbCur='',fbResolve=null;
function fbOpen(start){return new Promise(res=>{fbResolve=res;$('fbModal').classList.add('open');fbLoad(start||'');});}
function fbDone(val){$('fbModal').classList.remove('open');if(fbResolve){const r=fbResolve;fbResolve=null;r(val);}}
async function fbLoad(path){try{const d=await(await fetch('/api/browse?path='+encodeURIComponent(path||''),{cache:'no-store'})).json();
 fbCur=d.path||'';$('fbPath').value=fbCur;const box=$('fbList');box.innerHTML='';
 if(d.parent&&d.parent!==fbCur){const up=document.createElement('div');up.className='fbrow';up.textContent='⬆ .. (lên cấp trên)';up.onclick=()=>fbLoad(d.parent);box.appendChild(up);}
 (d.drives||[]).forEach(dr=>{const r=document.createElement('div');r.className='fbrow';r.textContent='💽 '+dr;r.onclick=()=>fbLoad(dr);box.appendChild(r);});
 (d.dirs||[]).forEach(nm=>{const r=document.createElement('div');r.className='fbrow';r.textContent='📁 '+nm;r.onclick=()=>fbLoad((fbCur.endsWith('/')?fbCur:fbCur+'/')+nm);box.appendChild(r);});
 if(d.error){const e=document.createElement('div');e.className='fbrow';e.style.color='#f7a5a5';e.textContent='⚠ '+d.error;box.appendChild(e);}
 if(!(d.dirs||[]).length&&!(d.drives||[]).length&&!d.error){const em=document.createElement('div');em.className='fbrow';em.style.color='var(--muted)';em.textContent='(không có thư mục con)';box.appendChild(em);}}
 catch(e){}}
// Bảng tổng quan theo ruộng (đọc /api/overview mỗi lần mở thẻ).
let ovTok=0;
async function loadOverview(){const tok=++ovTok;$('ovNote').textContent='⏳ đang đếm…';
 let d=null;try{d=await(await fetch('/api/overview',{cache:'no-store'})).json();}catch(e){d=null;}
 if(tok!==ovTok)return;
 if(!d){$('ovNote').textContent='⚠ không đọc được tổng quan';return;}
 const n=v=>Number(v).toLocaleString('vi-VN'),body=$('ovTable').tBodies[0],foot=$('ovTable').tFoot;
 const flw=f=>Object.keys(f).sort().map(r=>r+': '+Object.entries(f[r]).map(([s,c])=>c+' '+({done:'xong',doing:'đang',todo:'chưa'}[s]||s)).join(' · ')).join(' | ')||'—';
 body.innerHTML='';
 d.fields.forEach(r=>{const tr=document.createElement('tr');
  [r.field,n(r.images),n(r.labeled),n(r.empty),n(r.crowns),flw(r.flower)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td);});
  body.appendChild(tr);});
 foot.innerHTML='';const tr=document.createElement('tr');
 ['Tổng',n(d.total.images),n(d.total.labeled),n(d.total.empty),n(d.total.crowns),''].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td);});
 foot.appendChild(tr);
 $('ovNote').textContent='Không cây = ảnh đã xem và xác nhận không có tán (xuất như ảnh nền).'+(d.flower_error?' ⚠ nhãn hoa: '+d.flower_error:'');
 const box=$('ovChecks');box.innerHTML='';
 if(!d.checks.length){const s=document.createElement('span');s.className='ok';s.textContent='✔ Không tán nào có polygon đổi sau khi đã gán hoa.';box.appendChild(s);}
 d.checks.forEach(c=>{const s=document.createElement('span');s.className='warn';s.textContent='⚠ '+c.image+' · tán #'+(c.crown+1)+' · vòng '+c.round+': '+c.problem;box.appendChild(s);});}
tabRegister({id:'data',inspTitle:'Dữ liệu',help:[],enter(){$('hint').textContent='Dữ liệu · tổng quan theo ruộng và xuất bộ dữ liệu';renderStats();loadOverview();}});
// Xuất bộ nhãn hoa (thẻ Dữ liệu).
async function flExport(body){const info=$('flExpInfo');info.textContent='⏳ đang xuất…';
 let r=null;try{r=await(await fetch('/api/fl/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();}catch(e){r={ok:false,error:String(e)};}
 if(r.exists){info.textContent='';askConfirm({title:'Thư mục "'+body.name+'" đã tồn tại',ok:'Ghi đè',danger:true,text:'Ghi đè toàn bộ nội dung cũ của '+r.dir+'?'},
  ()=>flExport(Object.assign({},body,{overwrite:true})));return;}
 if(!r.ok){info.textContent='⚠ '+(r.error||'Lỗi xuất');return;}
 const s=r.splits||{};
 info.textContent='✔ train '+((s.train||{}).images||0)+' ảnh / '+((s.train||{}).crowns||0)+' tán · test '+((s.test||{}).images||0)+' ảnh / '+((s.test||{}).crowns||0)+' tán → '+r.dir+
  ((r.problems||[]).length?(' · ⚠ '+r.problems.length+' ảnh có vấn đề'):'');}
async function flExportStart(){const name=($('flExpName').value||'').trim();
 if(!name||/[\\/:*?"<>|]/.test(name)){$('flExpInfo').textContent='⚠ Tên bộ dữ liệu không hợp lệ.';return;}
 const out_dir=await fbOpen(expDirStart);if(!out_dir)return;expDirStart=out_dir;
 flExport({name:name,out_dir:out_dir,include_auto:$('flExpAuto').checked,overwrite:false});}
