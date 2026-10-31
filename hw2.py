# hw2.py
import os
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy import special
import my_library.analyze as analyze
import my_library.mtf_analysis as mtf
import my_library.image_metadata as img
import my_library.generate as gen
import my_library.linear_light as ll

if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    SPEED_UP = True
    TEST = False
    SHOW = True

    camera_path = 'images/iPhone16/'
    calib_path = 'all_calib_images/'
    lightroom = 'Lightroom/'

    slanted_path = f'{camera_path}{lightroom}hw2_slanted/'
    good_slanted_dir = f'{slanted_path}good_lighting/'
    bad_slanted_dir  = f'{slanted_path}bad_lighting/'

    output = 'output/hw2/'
    if TEST:
        calib_path = 'test/'

    calib_imgs_path = f'{camera_path}{calib_path}'

    # straight_lines_img = f'{calib_imgs_path}D1044335-EB95-4F60-8709-7B942E0B6825.DNG'
    straight_lines_img = f'{calib_imgs_path}3FB78461-4AFA-46EE-834B-9E01675900C4.DNG'
    # straight_lines_img = f'{calib_imgs_path}A56046EA-FEEA-4284-A51A-2DAF03198E01.DNG'
    # straight_lines_img = f'{calib_imgs_path}3D244010-6AE0-40AD-8050-317C858BC691.DNG'


    # K, distortion, corners = analyze.calibrate_camera(calib_imgs_path, oneoff=straight_lines_img, camera=cam)

    # original, undistorted = analyze.undistort(straight_lines_img, K, distortion, corners, cam)
    sigma = 1.0
    gen_mtf_curve = None

    # Problem 7
    problem_num = "07"
    print(f"\n**** Problem {problem_num} ****")
    system_mtf_in_um, predicted_mtf50_in_um = mtf.main(output)

    if not SPEED_UP:
        # Problem 8
        problem_num = "08"
        print(f"\n**** Problem {problem_num} ****")
        predicted_mtf50_in_pixels = predicted_mtf50_in_um * mtf.PITCH_UM * 2 # multiply by 2 since we are using the green-plane
        print(f"Predicted system MTF50 is:\n\t{predicted_mtf50_in_um:.4f} cycles/um\n\t{predicted_mtf50_in_um * 1000:.1f}  cycles/mm\n\t{predicted_mtf50_in_pixels:.4f} cycles per green pixel")


        # Problem 9
        problem_num = "09"
        print(f"\n**** Problem {problem_num} ****")
        theta = 5.0
        super_sampling = 8
        rnd_seed = np.random.randint(0, 2**32 - 1)
        print(f"Random seed for this run: {rnd_seed}")
        generated_slanted_edge = gen.capture_edge(sigma=sigma, angle_deg=theta, h=256, w=128, exposure=0.55, seed=rnd_seed, noisy=False, supersample_scale=super_sampling)
        if super_sampling > 1:
            # generated_slanted_edge = generated_slanted_edge.reshape(generated_slanted_edge.shape[0]//super_sampling, super_sampling, generated_slanted_edge.shape[1]//super_sampling, super_sampling).mean(axis=(1, 3)) # linear interpolation
            # Point-sample down by a factor of super_sampling
            generated_slanted_edge = generated_slanted_edge[super_sampling//2::super_sampling, super_sampling//2::super_sampling]  # point-sample at the center of each block
        print(f"Generated{' and sampled' if super_sampling > 1 else ' '} slanted edge --\n\tShape: {generated_slanted_edge.shape[1]} x {generated_slanted_edge.shape[0]} px\n\tRange in Linear Light: {generated_slanted_edge.min():.3f} to {generated_slanted_edge.max():.3f}")
        print(f"We expect to measure a system MTF50 = {mtf.analytic_gaussian_mtf50(sigma=sigma):.4f} cycles per pixel")

        '''
        Double check that the generated_slanted_edge matches an edge that has been captured by a lens
        with a Gaussian PSF with standard deviation of sigma.  This can be done by using the step response
        of a Gaussian filter, a normal CDF.
        1. Find d for each pixel, where d = (x_j - x_c - (y_i - y_c) * tan(theta)) * cos(theta)
            where x_c and y_c are width//2 and height//2, respectively.
        2. Compute the expected value of the step response at each pixel position, which is the CDF
        of a Gaussian with standard deviation sigma.  Normally this would be shifted by half a pixel to
            match the point-sampled edge, (i+0.5, j+0.5) but this didn't work out well in practice.
        '''
        x_c = generated_slanted_edge.shape[1] / 2
        y_c = generated_slanted_edge.shape[0] / 2
        theta_in_rads = np.deg2rad(theta)
        d = np.zeros_like(generated_slanted_edge)
        for i in range(generated_slanted_edge.shape[0]): #rows first
            for j in range(generated_slanted_edge.shape[1]): #cols second
                # d[i, j] = (j + 0.5 - x_c - (i + 0.5 - y_c) * np.tan(theta_in_rads)) * np.cos(theta_in_rads)
                d[i, j] = (j - x_c - (i - y_c) * np.tan(theta_in_rads)) * np.cos(theta_in_rads)
        expected_edge_cdf = 0.5 * (1 + special.erf(d / (np.sqrt(2) * sigma))) # Gaussian CDF with standard deviation sigma, shifted by half a pixel to match the point-sampled edge
        if not np.allclose(generated_slanted_edge, expected_edge_cdf, atol=1e-2):
            print(f"WARNING: Generated slanted edge does not match expected edge from Gaussian PSF with sigma = {sigma}.")

        fig, ax = plt.subplots(figsize=(3.2, 5))
        ax.imshow(generated_slanted_edge, cmap="gray")
        ax.contour(d, levels=[0], colors="red", linewidths=1.5, linestyles="--", label="Analytic Edge Center")
        ax.contour(generated_slanted_edge, levels=[0.5], colors="blue", linewidths=1.5, linestyles="--", label="Generated image crosses half brightness")
        ax.contour(d, levels=[-sigma, sigma], colors="green", linewidths=1.5, linestyles="--", label="Analytic Edge +/- sigma")
        # TODO fix the legend to show the correct labels for each contour line.
        # Could build the entries with Line2D proxy handles, or use the contour set's legend_elements() method to get the handles and labels for the legend.

        # # plot the line that is the expected_edge_cdf
        # '''
        # Note: This CDF will look reversed and falling.  This is correct and because the y-axis is flipped
        # and ranges from top to bottom instead of the traditional bottom to top.  The CDF is actually correct.
        # '''
        # ax.plot(np.arange(generated_slanted_edge.shape[1]), expected_edge[generated_slanted_edge.shape[0]//2, :] * generated_slanted_edge.shape[0], color="red", lw=1.5, label="Expected Edge (Gaussian CDF)")
        ax.set_title(f"Generated{' and sampled' if super_sampling > 1 else ' '} Slanted Edge", fontsize=9.5)
        ax.xaxis.tick_top()
        ax.legend()
        plt.tight_layout()
        if SHOW:
            plt.show()
        fig.savefig(f"{output}/09_generated_slanted_edge.png")


        # Problem 10
        problem_num = "10"
        print(f"\n**** Problem {problem_num} ****")
        freq, gen_mtf_curve, esf_x, esf, lsf, fitted, slope, intercept = mtf.slanted_edge_mtf(generated_slanted_edge)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

        axes[0].plot(esf_x, esf, ".", ms=2.5, color="blue")
        axes[0].set_xlim(-8, 8); axes[0].set_title("Edge spread function")
        axes[0].set_xlabel("distance from edge (px)")

        axes[1].plot(esf_x, np.gradient(esf, esf_x), color="green", lw=1.5)
        axes[1].set_xlim(-8, 8); axes[1].set_title("Line spread function ($\\frac{\\mathrm{\\delta}\\text{ESF}}{\\mathrm{\\delta}}$)")
        axes[1].set_xlabel("distance from edge (px)")

        axes[2].plot(freq, gen_mtf_curve, lw=2, color="forestgreen", label="measured from generated slanted edge")
        axes[2].plot(freq, mtf.get_analytic_mtf(freq, sigma=sigma), "--", lw=1.5,
                    color="orange", label="analytic")
        axes[2].axhline(0.5, color="lightgray", ls=":")
        axes[2].axvline(0.5, color="#555555", ls="--", lw=1.2)
        axes[2].text(0.51, 0.9, "Nyquist", fontsize=8.5, color="#555555")
        axes[2].set_xlim(0, 0.75); axes[2].set_ylim(0, 1.05)
        axes[2].set_xlabel("cycles per pixel"); axes[2].set_title(r"MTF ($\frac{|\mathcal{F}(LSF)|}{\mathrm{H}(0,0)}$)")
        axes[2].legend(fontsize=9)
        plt.tight_layout()
        if SHOW:
            plt.show()
        fig.savefig(f"{output}/10_ESF_LSF_MTF.png")

        # Problem 11
        problem_num = "11"
        print(f"\n**** Problem {problem_num} ****")
        # Print out the measured MTF50 from the generated slanted edge, and compare to the analytic MTF50.
        # And the percent error between the two.
        measured_mtf50 = mtf.mtf50(freq, gen_mtf_curve)
        analytic_mtf50 = mtf.analytic_gaussian_mtf50(sigma)
        percent_error = (measured_mtf50 - analytic_mtf50) / analytic_mtf50 * 100
        print(f"Measured MTF50 from generated slanted edge: {measured_mtf50:.4f} cycles per pixel") # are these green plane pixels? 
        print(f"Analytic MTF50 from Gaussian PSF with sigma = {sigma}: {analytic_mtf50:.4f} cycles per pixel") # are these green plane pixels?
        print(f"Percent error: {percent_error:.2f}%")

    raw_slant_paths = analyze._list_raw_files(good_slanted_dir)

    if not SPEED_UP:
        # Problem 12
        problem_num = "12"
        print(f"\n**** Problem {problem_num} ****")
        analyze.contact_sheet(raw_slant_paths, camera=cam, output=output, question=problem_num)

        # Problem 13
        problem_num = "13"
        print(f"\n**** Problem {problem_num} ****")
        img.assert_one_camera(raw_slant_paths)


    # Problem 14
    problem_num = "14"
    print(f"\n**** Problem {problem_num} ****")
    print("Extracting green plane from the Bayer mosaic...")
    with rawpy.imread(str(raw_slant_paths[0])) as image:
        green_img = ll.green_plane_from_raw(image, which_grn=0)
        #plot the green_img to verify that it looks like a slanted edge
        fig, ax = plt.subplots(figsize=(6, 5))
        # Normally this would go through gamma correction before being plotted, but I don't think
        # we had learned about that at this point...  So plot the linear light instead.
        ax.imshow(green_img, cmap="gray")
        ax.set_title(f"Green plane from {os.path.basename(raw_slant_paths[0])}", fontsize=9.5)
        ax.xaxis.tick_top()
        plt.tight_layout()
        if SHOW:
            plt.show()
        fig.savefig(f"{output}/14_green_plane.png")


    # Problem 15
    problem_num = "15"
    print(f"\n**** Problem {problem_num} ****")
    # measure the MTF50 on the central square of each capture in the green plane, and report the
    # mean and standard deviation of the MTF50 values in both um and cycles per green-plane pixel.
    mtf50_values_um = []
    mtf50_values_px = []
    mtf_curves = []
    for path in raw_slant_paths:
        with rawpy.imread(str(path)) as image:
            basename = os.path.basename(path).split('.')[0]
            green_img = ll.green_plane_from_raw(image, which_grn=0)
            # Get a central square of the green_img, 256x100 pixels, centered on the tilted square in the middle.
            # Take the square starting at row 600 and column 1120
            central_square = green_img[600:856, 1120:1220]
            #plot just the first one to confirm we have an edge
            if path == raw_slant_paths[0]:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(central_square, cmap="gray")
                ax.set_title(f"Central square of green plane from {basename}", fontsize=9.5)
                ax.xaxis.tick_top()
                plt.tight_layout()
                if SHOW:
                    plt.show()
                fig.savefig(f"{output}/15_central_square_edge_of_{basename}.png")
            freq, mtf_curve, esf_x, esf, lsf, fitted, slope, intercept = mtf.slanted_edge_mtf(central_square, real_data=True)
            mtf50_in_pixels = mtf.mtf50(freq, mtf_curve) # this si already in cycles per green-plane pixel
            # print the mtf50 in um and cycles per green-plane pixel
            mtf50_in_um = mtf50_in_pixels / (2 * mtf.PITCH_UM) # divide by 2 because we are using the green-plane, and multiply by pixel pitch in um to get cycles per um
            mtf50_values_um.append(mtf50_in_um)
            mtf50_values_px.append(mtf50_in_pixels)
            mtf_curves.append(mtf_curve)
            print(f"MTF50 for {basename}: {mtf50_in_um:.4f} cycles per um or {mtf50_in_pixels:.4f} cycles per green-plane pixel")

            # Plot the ESF, LSF, and MTF and save by the basename of the raw file
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

            axes[0].plot(esf_x, esf, ".", ms=2.5, color="blue")
            axes[0].set_xlim(-8, 8); axes[0].set_title("Edge spread function")
            axes[0].set_xlabel("distance from edge (px)")

            axes[1].plot(esf_x, np.gradient(esf, esf_x), color="green", lw=1.5)
            axes[1].set_xlim(-8, 8); axes[1].set_title("Line spread function ($\\frac{\\mathrm{\\delta}\\text{ESF}}{\\mathrm{\\delta}}$)")
            axes[1].set_xlabel("distance from edge (px)")
            axes[2].plot(freq, mtf_curve, ':', lw=1, color="red", label=f"measured from {basename}")
            if gen_mtf_curve:
                axes[2].plot(freq, gen_mtf_curve, lw=2, color="forestgreen", label="measured from generated slanted edge")
            axes[2].plot(freq, mtf.get_analytic_mtf(freq, sigma=sigma), "--", lw=1.5,
                        color="orange", label="analytic")
            axes[2].axhline(0.5, color="lightgray", ls=":")
            axes[2].axvline(0.5, color="#555555", ls="--", lw=1.2)
            axes[2].text(0.51, 0.9, "Nyquist", fontsize=8.5, color="#555555")
            axes[2].set_xlim(0, 0.75); axes[2].set_ylim(0, 1.05)
            axes[2].set_xlabel("cycles per green-plane pixel"); axes[2].set_title(r"MTF ($\frac{|\mathcal{F}(LSF)|}{\mathrm{H}(0,0)}$)")
            axes[2].legend(fontsize=9)
            fig.suptitle(f"ESF, LSF, and MTF for {basename}", fontsize=9.5)
            # fig.suptitle(f"Measured MTF50 = {mtf50_in_um:.4f} cycles per um or {mtf50_in_pixels:.4f} cycles per green-plane pixel", fontsize=9)
            plt.tight_layout()
            if SHOW:
                plt.show()
            fig.savefig(f"{output}/15_ESF_LSF_MTF_{basename}.png")
    # Make an overlay plot of all the mtf_curves
    cmap = plt.get_cmap("tab20")
    colors = cmap.colors
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, mtf_curve in enumerate(mtf_curves):
        ax.plot(freq, mtf_curve[:len(freq)], lw=1.5, label=f"{os.path.basename(raw_slant_paths[i]).split('.')[0]}", color=colors[i])
    if gen_mtf_curve:
        ax.plot(freq, gen_mtf_curve, lw=2, color="forestgreen", label="measured from generated slanted edge")
    # plot the system_mtf from the analytical MTF of the system
    # convert system_mtf from cycles per um to cycles per green-plane pixel by multiplying by 2 * mtf.PITCH_UM
    system_mtf_in_px = system_mtf_in_um * (2 * mtf.PITCH_UM)
    ax.plot(freq, system_mtf_in_px[:len(freq)], lw=2, color="black", label="system MTF from Problem 7")

    # ax.plot(freq, mtf.get_analytic_mtf(freq, sigma=sigma), "--", lw=1.5,
    #                         color="orange", label="analytic")

    ax.set_xlabel("cycles per green-plane pixel")
    ax.set_ylabel("MTF")
    ax.set_title("Overlay of MTF Curves")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if SHOW:
        plt.show()
    fig.savefig(f"{output}/15_MTF_Overlay.png")

    print(f"\nMean MTF50: {np.mean(mtf50_values_um):.4f} cycles per um or {np.mean(mtf50_values_px):.4f} cycles per green-plane pixel")
    print(f"Standard deviation: {np.std(mtf50_values_um, ddof=1):.4f} cycles per um or {np.std(mtf50_values_px, ddof=1):.4f} cycles per green-plane pixel")