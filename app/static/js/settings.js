// settings.js — nguồn dữ liệu, model SAM, nơi lưu nhãn.
async function loadFolders(){try{const d=await(await fetch('/api/folders',{cache:'no-store'})).json();
 curFolder=d.current||'';const s=$('folder');s.innerHTML='';
 const o0=document.createElement('option');o0.value='';o0.textContent='📁 Tất cả các field';s.appendChild(o0);
 (d.folders||[]).forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent='📁 '+f;s.appendChild(o);});
 s.value=curFolder;}catch(e){}}
async function loadModels(){const s=$('modelSel');if(!s)return;
 try{const d=await(await fetch('/api/models',{cache:'no-store'})).json();s.innerHTML='';
  (d.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=(m.variant==='sam3'?'✦3 ':'✦ ')+m.name;if(m.id===d.current)o.selected=true;s.appendChild(o);});
  if(!(d.models||[]).length){const o=document.createElement('option');o.textContent='(không có model trong weights/)';s.appendChild(o);}}
 catch(e){}}
// Nơi autosave nhãn là cố định theo configs/config.yaml — chỉ đọc để hiện lên
// tooltip chỉ báo lưu. Nơi xuất thì hỏi lại mỗi lần, đây chỉ là thư mục mở đầu.
let saveDirStart='',expDirStart='';
async function loadPaths(){try{const d=await(await fetch('/api/paths',{cache:'no-store'})).json();
 saveDirStart=d.out_dir||'';expDirStart=d.export_dir||'';
 svst('','','Nhãn lưu tại: '+(saveDirStart||'(mặc định)'));}catch(e){}}
