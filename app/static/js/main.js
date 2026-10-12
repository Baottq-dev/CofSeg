// main.js — gắn sự kiện và khởi động. Nạp CUỐI CÙNG: mọi hàm và biến của các
// file trước đã có, nên thứ tự các lệnh ở đây đúng như thứ tự trong bản gốc.
loadFlowerPrefs();
[['m-click','sam-click'],['m-box','sam-box'],['m-draw','draw'],['m-sel','select']].forEach(a=>$(a[0]).onclick=()=>setMode(a[1]));
$('addCls').onclick=autoAll;$('autoAll').onclick=autoAll;
$('satMax').oninput=()=>{satMax=parseInt($('satMax').value)||50;};
$('valMin').oninput=()=>{valMin=parseInt($('valMin').value)||180;};
// Đổi phương pháp hiển thị: vùng nào đã có số của phương pháp mới thì đổi ngay
// (nhãn auto đổi theo, nhãn tay giữ nguyên); vùng chưa có thì gọi server tính.
$('fmethod').onchange=()=>{fmethod=$('fmethod').value;saveFlowerPrefs();syncFlowerUI();
 if(!inst.length){draw();return;}
 pushHist();const miss=[];
 inst.forEach((r,i)=>{const v=r.pcts?r.pcts[fmethod]:null;
  // Số Otsu đã lưu có thể tính bằng kênh khác kênh đang chọn -> tính lại thay vì hiện số cũ.
  const stale=(fmethod==='otsu'&&r.od&&r.od.channel&&r.od.channel!==otsuP.channel);
  if(v==null||stale){miss.push(i);return;}r.pct=v;if(!r.manual)r.cls=densOf(v);});
 persist();renderRegions();renderClasses();draw();
 if(miss.length)computeFlower(miss,false);};
$('ochan').onchange=()=>{otsuP.channel=$('ochan').value;saveFlowerPrefs();$('sMaxRow').hidden=(otsuP.channel!=='white');};
$('sMax').oninput=()=>{otsuP.s_max=parseInt($('sMax').value)||0;saveFlowerPrefs();};
$('classSMax').oninput=()=>{otsuP.class_s_max=parseFloat($('classSMax').value)||0;saveFlowerPrefs();};
$('sepMin').oninput=()=>{otsuP.sep_min=parseFloat($('sepMin').value)||0;saveFlowerPrefs();};
$('areaMin').oninput=()=>{otsuP.area_min=parseInt($('areaMin').value)||0;saveFlowerPrefs();};
$('areaMax').oninput=()=>{otsuP.area_max=parseInt($('areaMax').value)||0;saveFlowerPrefs();};
$('elongMax').oninput=()=>{otsuP.elong_max=parseFloat($('elongMax').value)||0;saveFlowerPrefs();};
$('clipRule').onchange=()=>{otsuP.clip_rule=$('clipRule').checked;saveFlowerPrefs();};
$('showMask').onchange=()=>{showMask=$('showMask').checked;saveFlowerPrefs();scheduleMask();};
syncFlowerUI();
$('flowerOn').onclick=()=>{flowerOn=!flowerOn;saveFlowerPrefs();syncFlowerOn();renderRegions();scheduleMask();draw();};
syncFlowerOn();
$('applyFolder').onclick=async()=>{flush();
 const custom=$('folderPath').value.trim();const folder=custom||$('folder').value;
 try{const r=await(await fetch('/api/set_root',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder})})).json();
  if(!r.ok){alert(r.error||'Không đổi được thư mục');return;}
  store={};current=null;pathFilter='';await loadFolders();await loadImgs();}catch(e){alert('Lỗi đổi thư mục: '+e);}};
$('folder').onchange=()=>$('applyFolder').click();
$('search').oninput=renderThumbs;
$('save').onclick=openExport;
$('expClose').onclick=$('expCancel').onclick=()=>$('expModal').classList.remove('open');
$('expModal').onclick=e=>{if(e.target===$('expModal'))$('expModal').classList.remove('open');};
document.querySelectorAll('input[name=exsc]').forEach(r=>r.onchange=syncSplit);
['fmCoco','fmYolo','fmMask','fmInst'].forEach(id=>{const el=$(id);
 el.onchange=()=>el.closest('.exrow').classList.toggle('on',el.checked);});
$('spMode').onchange=syncSplit;$('spVal').oninput=syncSplit;$('spTest').oninput=syncSplit;
$('expGo').onclick=async()=>{
 const fmts=[];if($('fmCoco').checked)fmts.push('coco');if($('fmYolo').checked)fmts.push('yolo');
 if($('fmMask').checked)fmts.push('masks');if($('fmInst').checked)fmts.push('masks_instance');
 if(!fmts.length){$('expInfo').textContent='⚠ Chọn ít nhất một định dạng.';return;}
 const name=($('expName').value||'').trim();
 if(!name||/[\\/:*?"<>|]/.test(name)){$('expInfo').textContent='⚠ Tên bộ dữ liệu không hợp lệ.';return;}
 const v=curScope(),p=await preview(v);
 if(p&&!p.images_labeled){$('expInfo').textContent='⚠ Phạm vi này chưa có ảnh nào được gán nhãn.';return;}
 const out_dir=await fbOpen(expDirStart);if(!out_dir)return;expDirStart=out_dir;
 const body={scope:scopeOf(v),formats:fmts,name:name,out_dir:out_dir,
  val_folders:foldersOf('val'),test_folders:foldersOf('test'),
  split_by:$('spMode').value,val_ratio:(+$('spVal').value||0)/100,
  test_ratio:(+$('spTest').value||0)/100,seed:+$('spSeed').value||0,overwrite:false};
 await doExport(body,p);};
if($('modelSel'))$('modelSel').onchange=async()=>{const id=$('modelSel').value;$('hint').textContent='⏳ Đang chuyển model: '+id+' …';
 try{const r=await(await fetch('/api/model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id})})).json();
  $('hint').textContent=r.ok?('✔ Đã chọn model: '+id+' — sẽ nạp khi bấm SAM lần đầu.'):('⚠ '+(r.error||'Lỗi đổi model'));syncSamName();}
 catch(e){$('hint').textContent='⚠ '+e;}};
$('noTree').onclick=()=>{if(!current||!ready||inst.length)return;emptyMark=!emptyMark;persist();$('count').textContent=countText();};
$('undo').onclick=undo;
$('clear').onclick=()=>{closeCrMenu();clearAll();};
$('crTidyAll').onclick=()=>{closeCrMenu();tidyAll();};
$('crMore').onclick=e=>{e.stopPropagation();const m=$('crMenu'),on=!m.classList.contains('open');m.classList.toggle('open',on);$('crMore').setAttribute('aria-expanded',on);};
function closeCrMenu(){$('crMenu').classList.remove('open');$('crMore').setAttribute('aria-expanded',false);}
document.addEventListener('click',e=>{if(!e.target.closest('.dd'))closeCrMenu();});
$('crTidy').onclick=()=>{if(sel>=0&&!editLocked()){tidySel(sel);renderRegions();draw();}};
$('crDel').onclick=deleteSel;
$('zin').onclick=()=>zoomAt(cv.width/2,cv.height/2,1.25);
$('zout').onclick=()=>zoomAt(cv.width/2,cv.height/2,0.8);
$('zfit').onclick=()=>resetView();
// --- nối dây ----------------------------------------------------------------
$('predRun').onchange=e=>{predRun=e.target.value;predFetch();};
$('predConf').oninput=e=>{predConf=(+e.target.value)/100;
 $('predConfLbl').textContent=predConf.toFixed(2).replace('.',',');predSchedule();};
$('predShowGt').onchange=e=>{predShowGt=e.target.checked;draw();};
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect();zoomAt(e.clientX-r.left,e.clientY-r.top,e.deltaY<0?1.1:1/1.1);},{passive:false});
cv.addEventListener('mousedown',async e=>{
 if(e.button===1){pan=[e.clientX,e.clientY,ox,oy];cv.style.cursor='grabbing';e.preventDefault();return;}
 if(wsMouse('down',e))return;
 const xy=toImg(e),x=xy[0],y=xy[1];
 if(mode==='sam-click'){pushHist();await samCall('/api/sam/click',{name:current,x:x,y:y});}
 else if(mode==='sam-box'){boxStart=[x,y];}
 else if(mode==='draw'){draft.push(cX(x),cY(y));draw();}
 else if(mode==='select'){
  // Chỉ vùng ĐANG HIỆN mới nhận chuột: vùng ẩn mà vẫn được chọn thì click gần
  // đỉnh vô hình là kéo méo nó, chuột phải là xoá đỉnh nó, không thấy gì hết.
  if(sel>=0&&inst[sel]&&inst[sel].vis){const k=hitVertex(inst[sel].poly,x,y);
   if(k>=0){if(e.button===2){pushHist();inst[sel].poly.splice(k,2);if(inst[sel].poly.length<6){inst.splice(sel,1);sel=-1;}persist();renderRegions();draw();return;}
    pushHist();
    // beginDrag chèn điểm CHỈ dưới bút và bám đúng đỉnh vừa bấm (bản cũ dò lại
    // bằng nearestVi theo vị trí con trỏ nên hay tóm nhầm đỉnh khác, hình giật
    // một cái lúc bắt đầu kéo).
    beginDrag(k>>1,x,y,e.shiftKey);return;}
   // Không trúng đỉnh nào nhưng đang rê trên cạnh -> biến đỉnh giả thành đỉnh
   // thật rồi kéo luôn, khỏi phải nhắm vào mấy đỉnh thưa.
   if(e.button===0&&ghost){pushHist();
    const Q=inst[sel].poly,n=Q.length/2,j=(ghost.i+1)%n;
    // Đã có đỉnh thật sát ngay đó (đỉnh bị ẩn vì handleIdx vẽ thưa) thì dùng
    // chính nó, đừng chèn một đỉnh gần như trùng.
    const da=Math.hypot(Q[ghost.i*2]-ghost.x,Q[ghost.i*2+1]-ghost.y)*scale;
    const db=Math.hypot(Q[j*2]-ghost.x,Q[j*2+1]-ghost.y)*scale;
    let vi;
    if(da<HANDLE_PX*0.5)vi=ghost.i;
    else if(db<HANDLE_PX*0.5)vi=j;
    else{Q.splice(ghost.i*2+2,0,ghost.x,ghost.y);vi=ghost.i+1;}
    beginDrag(vi,x,y,e.shiftKey);renderRegions();return;}}
  const ip=hitPolygon(x,y);
  if(ip>=0){sel=ip;dragPt=-1;tidySel(ip);renderRegions();draw();}
  else{sel=-1;pan=[e.clientX,e.clientY,ox,oy];cv.style.cursor='grabbing';renderRegions();draw();}}
 // Chế độ xem: kéo ở BẤT KỲ đâu cũng là di chuyển ảnh. Không có nhánh nào
 // chạm vào `inst` nên không có gì để khoá thêm.
 else if(mode==='view'){pan=[e.clientX,e.clientY,ox,oy];cv.style.cursor='grabbing';}});
cv.addEventListener('mousemove',e=>{
 if(pan){ox=pan[2]+(e.clientX-pan[0]);oy=pan[3]+(e.clientY-pan[1]);draw();return;}
 if(wsMouse('move',e))return;
 mouse=toImg(e);
 if(mode==='select'&&dragPt>=0&&sel>=0&&origPoly&&softD){const P=inst[sel].poly,dx=mouse[0]-dragStart[0],dy=mouse[1]-dragStart[1],N=origPoly.length/2;
  // Trọng số theo KHOẢNG CÁCH dọc biên (px màn hình), không theo số thứ tự đỉnh,
  // nên bút luôn rộng đúng INFLUENCE_PX dù đỉnh thưa hay dày.
  // Bán kính CỐ ĐỊNH, KHÔNG nở theo độ kéo. Bản nở làm số đỉnh bị kéo theo tăng
  // vọt (đo trên tán thật ở zoom 1200%: kéo 10px động 15% đường biên, kéo 100px
  // động 62%) nên không nắn cục bộ được nữa. Nó vốn để chống gấp mép, nhưng việc
  // đó nay do untanglePoly lo lúc thả chuột — bỏ đi số đỉnh chỉ đổi 153->163 và
  // chỗ tự cắt lưu xuống file vẫn bằng 0.
  for(let i=0;i<N;i++){let w=0;
   if(!dragSoft)w=(i===dragVi)?1:0;                  // Shift = chỉ 1 đỉnh
   else{const dd=softD[i];
    if(isFinite(dd)&&dd<INFLUENCE_PX)
     w=0.5*(1+Math.cos(Math.PI*(dd/INFLUENCE_PX)));}
   if(w<=0)continue;   // R cố định -> tập đỉnh chịu ảnh hưởng không đổi cả nhát
   P[i*2]=cX(origPoly[i*2]+dx*w);P[i*2+1]=cY(origPoly[i*2+1]+dy*w);}
  draw();}
 else if(mode==='sam-box'&&boxStart){draw();}
 else if(mode==='draw'&&draft.length){draw();}
 else if(mode==='select'){const hv=(sel>=0&&inst[sel]&&inst[sel].vis)?hitVertex(inst[sel].poly,mouse[0],mouse[1]):-1;
  // Đỉnh thật được ưu tiên; chỉ khi trượt hết mới tìm điểm trên cạnh.
  const g=(hv<0&&sel>=0&&inst[sel]&&inst[sel].vis)?
   hitEdge(inst[sel].poly,mouse[0],mouse[1]):null;
  const cur=(hv>=0)?'move':(g?'copy':((hitPolygon(mouse[0],mouse[1])>=0)?'pointer':'grab'));
  if(cv.style.cursor!==cur)cv.style.cursor=cur;
  const moved=(!g!==!ghost)||(g&&ghost&&(g.x!==ghost.x||g.y!==ghost.y));
  if(hv!==hovPt||moved){hovPt=hv;ghost=g;draw();}}
 else if(mode==='view'){const h=predPick(mouse[0],mouse[1]);
  const doi=(!h!==!predHover)||(h&&predHover&&(h.kind!==predHover.kind||h.i!==predHover.i));
  if(doi){predHover=h;cv.style.cursor=h?'help':'grab';predHint();draw();}}});
// Nhả chuột bắt trên WINDOW, không phải canvas: kéo đỉnh ra tới panel bên rồi
// nhả thì canvas không nhận mouseup, dragPt/origPoly còn nguyên, rê chuột về là
// cả cụm đỉnh chạy theo con trỏ. Với vùng đã ẩn thì không thấy gì cho tới khi
// hiện lại và hình đã méo. Pan chuột giữa cũng kẹt y hệt.
window.addEventListener('mouseup',async e=>{
 // Thẻ Tán ở chế độ Xem: bấm (không kéo) lên một tán là chọn để xem chi tiết.
 if(pan){const click=Math.abs(e.clientX-pan[0])<3&&Math.abs(e.clientY-pan[1])<3;pan=null;cv.style.cursor=(mode==='select'||mode==='view')?'grab':'crosshair';
  if(wsMouse('panEnd',e))return;
  if(click&&wsActive==='crown'&&mode==='view'&&!predOn){const xy=toImg(e);sel=hitPolygon(xy[0],xy[1]);renderRegions();draw();}
  return;}
 if(wsMouse('up',e))return;
 if(mode==='sam-box'&&boxStart){const xy=toImg(e),x=xy[0],y=xy[1],b=[cX(Math.min(boxStart[0],x)),cY(Math.min(boxStart[1],y)),cX(Math.max(boxStart[0],x)),cY(Math.max(boxStart[1],y))];boxStart=null;
  if(Math.abs(b[2]-b[0])>5&&Math.abs(b[3]-b[1])>5){pushHist();await samCall('/api/sam/box',{name:current,box:b});}else draw();return;}
 if(dragPt>=0){dragPt=-1;dragVi=-1;origPoly=null;softD=null;
  cv.style.cursor='move';
  if(sel>=0&&inst[sel])inst[sel].poly=cleanPoly(inst[sel].poly);
  persist();renderRegions();draw();if(sel>=0)computeFlower([sel]);}});
cv.addEventListener('mouseleave',e=>{if(wsMouse('leave',e))return;if(!pan&&dragPt<0){hovPt=-1;ghost=null;
 if(predHover){predHover=null;predHint();}
 if(mode==='select')cv.style.cursor='crosshair';draw();}});
cv.addEventListener('dblclick',e=>{if(wsMouse('dblclick',e))return;if(mode==='draw'&&draft.length>=6){pushHist();const ni=addRegion(draft.slice());draft=[];sel=inst.length-1;refresh();computeFlower([ni]);}});
cv.addEventListener('contextmenu',e=>e.preventDefault());
window.addEventListener('keydown',e=>{
 if(e.key==='Escape'&&closeModals())return;
 if(typing(e))return;
 // Alt+1…4 theo vị trí phím (e.code), để bàn phím gõ tiếng Việt không đổi số.
 if(e.altKey&&/^Digit[1-4]$/.test(e.code)){e.preventDefault();switchWs(TAB_ORDER[+e.code.slice(5)-1]);return;}
 if(e.key==='PageDown'||e.key==='PageUp'){e.preventDefault();stepImg(e.key==='PageDown'?1:-1);return;}
 if(e.key==='?'){openHelp();return;}
 if(wsKey(e))return;
 if(wsActive==='data'||e.ctrlKey||e.metaKey||e.altKey)return;
 if(e.key==='+'||e.key==='=')zoomAt(cv.width/2,cv.height/2,1.25);
 if(e.key==='-')zoomAt(cv.width/2,cv.height/2,0.8);
 if(e.key==='0')resetView();});
window.addEventListener('resize',()=>{if(img.width){sizeCanvas();resetView();}});
window.addEventListener('beforeunload',beacon);
document.querySelectorAll('.resizer').forEach(rz=>{rz.addEventListener('mousedown',e=>{e.preventDefault();const el=$(rz.dataset.t),x0=e.clientX,w0=el.getBoundingClientRect().width,dir=rz.dataset.dir==='r'?-1:1;rz.classList.add('drag');document.body.style.userSelect='none';
 const mv=ev=>{const w=Math.max(150,Math.min(560,w0+dir*(ev.clientX-x0)));el.style.width=w+'px';if(img.width){sizeCanvas();draw();}};
 const up=()=>{rz.classList.remove('drag');document.body.style.userSelect='';window.removeEventListener('mousemove',mv);window.removeEventListener('mouseup',up);};
 window.addEventListener('mousemove',mv);window.addEventListener('mouseup',up);});});
$('fbGo').onclick=()=>fbLoad($('fbPath').value.trim());
$('fbPath').addEventListener('keydown',e=>{if(e.key==='Enter')fbLoad($('fbPath').value.trim());});
$('fbClose').onclick=()=>fbDone(null);
$('fbCancel').onclick=()=>fbDone(null);
$('fbModal').onclick=e=>{if(e.target===$('fbModal'))fbDone(null);};
$('fbPick').onclick=()=>fbDone(fbCur);
$('fbMk').onclick=async()=>{const nm=$('fbNew').value.trim();if(!nm)return;
 try{const r=await(await fetch('/api/mkdir',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:fbCur,name:nm})})).json();
  if(r.ok){$('fbNew').value='';fbLoad(fbCur);}else alert(r.error||'Không tạo được thư mục');}catch(e){alert(e);}};
// --- khung: thẻ, cột, điều hướng, cài đặt, bảng phím ------------------------------
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>switchWs(b.dataset.tab));
$('navClose').onclick=$('navOpen').onclick=()=>toggleCol('nav');
$('inspClose').onclick=$('inspOpen').onclick=()=>toggleCol('insp');
$('navField').onchange=()=>pickDir($('navField').value);
document.querySelectorAll('#navFilter button').forEach(b=>b.onclick=()=>setNavFilter(b.dataset.f));
$('navThumbs').onchange=()=>setNavThumbs($('navThumbs').checked);
$('veView').onclick=()=>setMode('view');
$('veEdit').onclick=askEdit;
$('btnHelp').onclick=openHelp;
$('btnSettings').onclick=openSettings;
$('setClose').onclick=()=>$('setModal').classList.remove('open');
$('setModal').onclick=e=>{if(e.target===$('setModal'))$('setModal').classList.remove('open');};
$('annotAdd').onclick=addAnnot;
// --- thẻ Hoa ------------------------------------------------------------------
document.querySelectorAll('#flTools button').forEach(b=>b.onclick=()=>setFlTool(b.dataset.tool));
$('flBrush').oninput=()=>setFlBrush(+$('flBrush').value);
$('flOpacity').oninput=()=>setFlOpacity(+$('flOpacity').value/100);
$('flLearn').onclick=flClassify;
$('flUndo').onclick=flUndo;
$('flDone').onclick=flFinish;
$('flNext').onclick=flNext;
$('flRound').onchange=()=>{flRound=$('flRound').value;lsSet('cofseg_flround',flRound);flCrown=-1;flLoadConfig().then(flLoadCrowns);};
document.querySelectorAll('.fllayers button').forEach(b=>b.onclick=()=>toggleFlLayer(b.dataset.layer));
window.addEventListener('keyup',e=>{if(e.key===' '&&flSpace){flSpace=false;if(flMouse)flCursor(flMouse[0],flMouse[1]);}});
$('annotNew').addEventListener('keydown',e=>{if(e.key==='Enter')addAnnot();});
$('helpClose').onclick=$('helpOk').onclick=()=>$('helpModal').classList.remove('open');
$('helpModal').onclick=e=>{if(e.target===$('helpModal'))$('helpModal').classList.remove('open');};
// Thẻ Tán mở ở chế độ Xem: nhãn tán đã chốt, không có cú bấm nào lỡ sửa được nó.
renderClasses();setMode('view');renderAnnot();setNavFilter(navFilter);$('navThumbs').checked=navThumbs;
switchWs(lsGet('cofseg_tab','crown'));loadModels().then(syncSamName);loadPaths();loadFolders().then(loadImgs);
