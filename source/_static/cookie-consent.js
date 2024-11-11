// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

document.addEventListener('DOMContentLoaded', function () {
    // Check if the user has already given consent
    if (localStorage.getItem('cookieConsent') !== 'true') {
        // Show the cookie consent banner
        showCookieConsent();
    }else{
        initializeGoogleAnalytics();
    }
});

function showCookieConsent() {
    var cookieConsent = document.getElementById('cookie-consent');

    // Display the cookie consent banner
    cookieConsent.style.display = 'block';

    // Add event listener to the "Got it!" link
    document.getElementById('cookie-accept').addEventListener('click', function () {
        // Set a localStorage flag to remember user's consent
        localStorage.setItem('cookieConsent', 'true');

        // Hide the cookie consent banner
        cookieConsent.style.display = 'none';

        // Add Google Analytics tracking code
        initializeGoogleAnalytics();
    });

    // Add event listener to the "Reject" link
    document.getElementById('cookie-reject').addEventListener('click', function () {
        // Set a localStorage flag to remember user's rejection
        localStorage.setItem('cookieRejected', 'true');

        // Hide the cookie consent banner
        cookieConsent.style.display = 'none';
    });
}

function initializeGoogleAnalytics() {
    console.log("Initializing Google Analytics...");

    // Add Google Analytics tracking code
    (function (i, s, o, g, r, a, m) {
        console.log("Executing self-invoking function...");

        i['GoogleAnalyticsObject'] = r;
        (i[r] =
            i[r] ||
            function () {
                (i[r].q = i[r].q || []).push(arguments);
                console.log("gtag function called with arguments:", arguments);
            }),
            (i[r].l = 1 * new Date());
        console.log("GoogleAnalyticsObject set to 'gtag' and timestamp added.");

        (a = s.createElement(o)), (m = s.getElementsByTagName(o)[0]);
        a.async = 1;
        a.src = g;
        m.parentNode.insertBefore(a, m);
        console.log("Google Analytics script tag created and inserted:", a);
    })(
        window,
        document,
        'script',
        'https://www.googletagmanager.com/gtag/js?id=G-JK8T9PJNL0',
        'gtag'
    );

    // Set consent status
    gtag('consent', 'update', {
            'ad_user_data': 'granted',
            'ad_storage': 'granted',
            'analytics_storage': 'granted'
    });
    console.log("Consent status updated.");

    gtag('js', new Date());
    console.log("gtag('js', new Date()) called.");

    gtag('config', 'G-JK8T9PJNL0');
    console.log("gtag('config', 'G-JK8T9PJNL0') called.");
}