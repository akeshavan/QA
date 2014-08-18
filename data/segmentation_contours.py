import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

def segmentation_checker(in_file,seg_file,mask_file=None,saveas=None,do_contour=True):
    """
    Shows x,y,z mosaics with outlined labels. Optionally save each image.
    """
    import numpy as np
    import nibabel as nib
    from matplotlib.pyplot import subplots,savefig
    from nipy.labs.viz import plot_anat    
   
    img1 = nib.load(in_file)
    data1,aff1 = img1.get_data(),img1.get_affine()
    if mask_file:
        mask = nib.load(mask_file).get_data()
        data= np.ma.array(data1,mask=mask==0)
    labels = nib.load(seg_file).get_data()
    if mask_file:
        labels = np.ma.array(labels,mask=labels==0)
    xminv,yminv,zminv = np.min(np.nonzero(labels),1)
    xmin,ymin,zmin,_ = np.dot(aff1, [xminv,yminv,zminv,1])
    xmaxv,ymaxv,zmaxv = np.max(np.nonzero(labels),1)
    xmax,ymax,zmax,_ = np.dot(aff1, [xmaxv,ymaxv,zmaxv,1])
    xshift,yshift,zshift,_ = np.dot(aff1,[xmaxv-xminv,ymaxv-yminv,zmaxv-zminv,1])
    
    bounds = lambda xmin,xmax:((xmax - xmin)*0.2 + xmin, xmax - (xmax - xmin)*0.2)
    
    fig,axs = subplots(3,3,figsize=(6,6))
    fig.subplots_adjust(hspace = -0.5,wspace=0.01)
    axs = axs.ravel()
    for i, x_coord in enumerate(np.linspace(*bounds(xmin,xmax),num=9)):
        sl = plot_anat(data,aff1,slicer="x",cut_coords=[x_coord],axes=axs[i],annotate=False)
        if do_contour: sl.contour_map(labels,aff1,levels=range(-50,50),colors="yellow")
        foo = axs[i]
        foo.set_adjustable('box-forced')
        
    if saveas:
        savefig(saveas+"_x.png")
        
    fig,axs = subplots(3,3,figsize=(6,6))
    fig.subplots_adjust(hspace = -0.25,wspace=0.01)
    #fig.subplots_adjust(hspace = -0.55,wspace=-0.3)
    axs = axs.ravel()
    bmin,bmax = bounds(ymin,ymax) 
    for i, coord in enumerate(np.linspace(bmin,bmax,num=9)):
        sl = plot_anat(data,aff1,slicer="y",cut_coords=[coord],axes=axs[i],annotate=False)
        if do_contour: sl.contour_map(labels,aff1,levels=range(-50,50),colors="yellow")
    if saveas:
        savefig(saveas+"_y.png")
        
    fig,axs = subplots(3,3,figsize=(6,6))
    fig.subplots_adjust(hspace = -0.01,wspace=0.01)
    #fig.subplots_adjust(hspace = -0.3,wspace=-0.3)
    axs = axs.ravel()
    bmin,bmax = bounds(zmin,zmax) 
    for i, coord in enumerate(np.linspace(bmin+15,bmax+10,num=9)):
        sl = plot_anat(data,aff1,slicer="z",cut_coords=[coord],axes=axs[i],annotate=False)
        if do_contour: sl.contour_map(labels,aff1,levels=range(-50,50),colors="yellow")
        
    if saveas:
        savefig(saveas+"_z.png")

if __name__=="__main__":
    import pandas as pd
    import os

    fs = pd.read_csv("../db/freesurfer_info.csv")
    ids = fs.fs_subject_id

    wf = pe.Workflow(name="images_QA")
    wf.base_dir = os.environ["SCRATCH_DIR"]
    seg = pe.Node(niu.Function(input_names=["in_file","seg_file","mask_file","saveas","do_contour"],
                               output_names=[],function=segmentation_checker),name="image_maker")
    for i in ids:
        folder = os.path.abspath(os.path.join("freesurfer",i))
        origfolder = os.path.abspath(os.path.join("orig",i))
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(origfolder):
            os.makedirs(origfolder)
        surfdir = os.path.join(os.environ["SUBJECTS_DIR"],i,"mri")
        ifile, sfile, mfile = os.path.join(surfdir,"orig.mgz"),os.path.join(surfdir,"aseg.mgz"),os.path.join(surfdir,"brainmask.mgz") 
        if os.path.exists(ifile) and os.path.exists(sfile) and os.path.exists(mfile):
            s1 = seg.clone("seg_%s"%i)
            s1.inputs.in_file = ifile
            s1.inputs.seg_file = sfile
            s1.inputs.mask_file = mfile
            s1.inputs.do_contour = True
            s1.inputs.saveas = os.path.join(folder,i)
            s2 = seg.clone("orig_%s"%i)
            s2.inputs.in_file = ifile
            s2.inputs.seg_file = sfile
            s2.inputs.mask_file = mfile
            s2.inputs.do_contour = False
            s2.inputs.saveas = os.path.join(origfolder,i)
            #segmentation_checker(ifile,sfile,mfile,os.path.join(folder,i))
            #segmentation_checker(ifile,sfile,mfile,os.path.join(origfolder,i),do_contours=False)
            wf.add_nodes([s1,s2])
            print "added",i  

    wf.run(plugin="SGE",plugin_args={"qsub_args":os.environ["PLUGIN_ARGS"]})

